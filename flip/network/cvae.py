import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from utils.network_utils import get_2d_sincos_pos_embed, HWPatchEmbed
from einops import rearrange
import torch.nn.functional as F

class ScaleDirectionCVAE(nn.Module):
    def __init__(
        self,
        hidden_size,
        vae_encoder_layers,
        vae_decoder_layers,
        num_heads,
        flow_horizon=16,
        obs_history=4,
        img_size=[128, 128],
        img_patch_size=8,
        language_original_dim=4096,
    ):
        super(ScaleDirectionCVAE, self).__init__()

        self.hidden_size = hidden_size
        self.vae_encoder_layers = vae_encoder_layers
        self.vae_decoder_layers = vae_decoder_layers
        self.num_heads = num_heads
        self.flow_horizon = flow_horizon
        self.obs_history = obs_history
        self.img_size = img_size
        self.img_patch_size = img_patch_size
        self.language_original_dim = language_original_dim

        self.language_encoder = nn.Linear(language_original_dim, hidden_size)   # encode [B, T, D_lang] -> [B, T, D]
        if self.img_size[0] == self.img_size[1]:    
            self.vision_encoder = PatchEmbed(
                img_size=self.img_size[0],
                patch_size=self.img_patch_size,
                in_chans=3 * obs_history,
                embed_dim=hidden_size,
            )
        else:
            self.vision_encoder = HWPatchEmbed(
                img_size=self.img_size,
                patch_size=self.img_patch_size,
                in_chans=3 * obs_history,
                embed_dim=hidden_size,
            )
        self.vision_decoder = nn.Linear(hidden_size, 3 * obs_history * self.img_patch_size ** 2, bias=True) # for aux loss

        self.query_point_encoder = nn.Linear(2, hidden_size)
        self.flow_encoder = nn.Linear(2*self.flow_horizon, hidden_size)
        self.flow_decoder = nn.Linear(hidden_size, self.flow_horizon * 2)   # for the flow output part of the scene encoder for aux loss
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # CVAE
        self.cvae_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
            ),
            num_layers=vae_encoder_layers,
        )

        self.cvae_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
            ),
            num_layers=vae_decoder_layers,
        )

        self.z_mu = nn.Linear(hidden_size, hidden_size)
        self.z_logvar = nn.Linear(hidden_size, hidden_size)

        self.scale_head = nn.Sequential(
            nn.Linear(hidden_size, 1*self.flow_horizon),
            nn.Softplus(),  # scale is non-negative.
        )

        self.direction_head = nn.Linear(hidden_size, 2*self.flow_horizon)
        
        # positional embedding
        self.img_pos_embedding = nn.Parameter(torch.randn(1, self.vision_encoder.num_patches, hidden_size), requires_grad=False)
        num_patches_h, num_patches_w = self.img_size[0] // self.img_patch_size, self.img_size[1] // self.img_patch_size
        img_embed = get_2d_sincos_pos_embed(hidden_size, (num_patches_h, num_patches_w))
        img_embed = rearrange(img_embed, 'n d -> () n d')
        self.img_pos_embedding.data.copy_(torch.from_numpy(img_embed))

    def get_embedding(self, obs_history, sentence_embedding):
        ''' obs_history: [B, T, C, H, W]
            sentence_embedding: [B, L, D_lang]
        '''
        language_embedding = self.language_encoder(sentence_embedding)

        vid = rearrange(obs_history, 'b t c h w -> b (t c) h w')
        img_patches = self.vision_encoder(vid) + self.img_pos_embedding    # [B, H//p*W//p, hidden_size]

        return img_patches, language_embedding

    def forward(self, gt_flow, obs_history, sentence_embedding):
        ''' gt_flow: [B, N, T, 2]
            obs_history: [B, T, C, H, W]
            sentence_embedding: [B, L, D_lang]
        '''
        B, N, _, _ = gt_flow.shape
        query_points = gt_flow[:, :, 0]
        padding_mask = torch.all(sentence_embedding == 0, dim=-1)

        img_patches, language_embedding = self.get_embedding(obs_history, sentence_embedding)

        flow_embedding = self.flow_encoder(rearrange(gt_flow, 'b n t d -> b n (t d)'))

        # encoder forward
        encoder_input = torch.cat([self.cls_token.repeat(B, 1, 1), flow_embedding, img_patches, language_embedding], dim=1)
        full_mask = torch.cat([torch.zeros(B, 1 + N + img_patches.shape[1], dtype=torch.bool, device=padding_mask.device), padding_mask], dim=1)
        encoder_output = self.cvae_encoder(encoder_input.transpose(0, 1), src_key_padding_mask=full_mask).transpose(0, 1)
        z_embedding = encoder_output[:, 0]
        img_output = self.vision_decoder(encoder_output[:, 1+N:1+N+img_patches.shape[1]])
        flow_output = self.flow_decoder(encoder_output[:, 1:1+N]).reshape(B, N, self.flow_horizon, 2)

        z_mu = self.z_mu(z_embedding)
        z_logvar = self.z_logvar(z_embedding)
        z_logvar = torch.clamp(z_logvar, min=-10, max=10)
        z_std = torch.exp(0.5 * z_logvar)
        z = torch.distributions.normal.Normal(z_mu, torch.exp(z_std)).rsample()

        # decoder forward
        q_point_embedding = self.query_point_encoder(query_points)
        decoder_input = torch.cat([z.unsqueeze(1), q_point_embedding, img_patches, language_embedding], dim=1)
        full_mask = torch.cat([torch.zeros(B, 1 + N + img_patches.shape[1], dtype=torch.bool, device=padding_mask.device), padding_mask], dim=1)
        decoder_output = self.cvae_decoder(decoder_input.transpose(0, 1), src_key_padding_mask=full_mask).transpose(0, 1)
        scale = self.scale_head(decoder_output[:, 1:1+N]).reshape(B, N, self.flow_horizon, 1)[:, :, :-1]  # T-1 steps
        direction = self.direction_head(decoder_output[:, 1:1+N]).reshape(B, N, self.flow_horizon, 2)[:, :, :-1]  # T-1 steps
        direction = F.normalize(direction, p=2, dim=-1)

        return scale, direction, z_mu, z_logvar, img_output, flow_output

    def inference(self, query_points, img_history, sentence_embedding, z):
        ''' z: [B, D]
            query_points: [B, N, 2]
            img_history: [B, T, C, H, W]
            sentence_embedding: [B, L, D_lang]
        '''
        B, N, _ = query_points.shape
        padding_mask = torch.all(sentence_embedding == 0, dim=-1)

        img_patches, language_embedding = self.get_embedding(img_history, sentence_embedding)

        q_point_embedding = self.query_point_encoder(query_points)
        decoder_input = torch.cat([z.unsqueeze(1), q_point_embedding, img_patches, language_embedding], dim=1)
        full_mask = torch.cat([torch.zeros(B, 1 + N + img_patches.shape[1], dtype=torch.bool, device=padding_mask.device), padding_mask], dim=1)
        decoder_output = self.cvae_decoder(decoder_input.transpose(0, 1), src_key_padding_mask=full_mask).transpose(0, 1)
        scale = self.scale_head(decoder_output[:, 1:1+N]).reshape(B, N, self.flow_horizon, 1)[:, :, :-1]
        direction = self.direction_head(decoder_output[:, 1:1+N]).reshape(B, N, self.flow_horizon, 2)[:, :, :-1]
        direction = F.normalize(direction, p=2, dim=-1)

        return scale, direction

    def _patchify(self, imgs):
        """
        imgs: (N, T, 3, H, W) or lantent (N, T, 4, H, W)
        x: (N, L, patch_size**2 * T * 3) or (N, L, T * 4)
        """

        N, T, C, img_H, img_W = imgs.shape
        p = self.vision_encoder.patch_size[0]
        assert img_H % p == 0 and img_W % p == 0

        h = img_H // p
        w = img_W // p
        x = imgs.reshape(shape=(imgs.shape[0], T, C, h, p, w, p))
        x = rearrange(x, "n t c h p w q -> n h w p q t c")
        x = rearrange(x, "n h w p q t c -> n (h w) (p q t c)")
        return x
    
    def img_recon_loss(self, img_output, img):
        '''args:
                img_embedding: [B, L, patch_size**2 * obs_history * 3]
                img: [B, T, 3, H, W]
        '''
        img_clone = img.clone()
        patched_img = self._patchify(img_clone)

        return F.mse_loss(img_output, patched_img)

    def flow_recon_loss(self, recon_flow, gt_flow):
        '''
            args:
                predicted_flow: [B, N, T, 2]
                gt_flow: [B, N, T, 2]
        '''

        return F.mse_loss(recon_flow, gt_flow)

    def reconstruct_flow(self, q_points, scale, direction):
        ''' reconstrut the flow given the predicted scale and direction
            args:
                q_points: [B, N, 2]
                scale: [B, N, 1]
                direction: [B, N, T-1, 2]
        '''
        B, N, T, _ = direction.shape
        T += 1

        flow = torch.zeros(B, N, T, 2).to(q_points.device)
        flow[:, :, 0] = q_points
        for i in range(1, self.flow_horizon):
            flow[:, :, i] = flow[:, :, i-1] + scale[:, :, i-1] * direction[:, :, i-1]

        return flow

    def kl_loss(self, z_mu, z_logvar, robust=True):
        # loss = 0.5 * torch.sum(z_mu ** 2 + z_std ** 2 - torch.log(z_std ** 2) - 1)
        loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar))
        if robust:
            loss = torch.sqrt(1 + loss ** 2) - 1

        return loss

    def direction_loss(self, pred, target, scale_weight=None, threshold=2., weight=100.):
        if scale_weight is not None:
            # add more weight on large moves
            loss = torch.sum((pred - target) ** 2, dim=-1)  # [B, N, T-1]
            new_weight = torch.ones_like(loss)
            new_weight[loss > threshold] = weight
            loss = torch.mean(new_weight * loss)
            return loss
        else:
            return F.mse_loss(pred, target)

    def scale_loss(self, pred, target, scale_weight=None, threshold=2., weight=100.):
        if scale_weight is not None:
            # add more weight on large moves
            loss = torch.sum((pred - target) ** 2, dim=-1)  # [B, N, T-1]
            new_weight = torch.ones_like(loss)
            new_weight[loss > threshold] = weight
            loss = torch.mean(new_weight * loss)
            return loss
        else:
            return torch.nn.functional.mse_loss(pred, target)

    def average_distance_error(self, gt_flow, scale, direction, scale_weight=None, threshold=1., weight=100., eval_mode=False):
        recon_flow = self.reconstruct_flow(gt_flow[:, :, 0].clone(), scale, direction)
        if scale_weight is not None:
            if eval_mode:   # only calculate the ADE for the frames with large moves
                mask = scale_weight[..., 0] > threshold
                loss = torch.sqrt(torch.sum((recon_flow - gt_flow) ** 2, dim=-1))[:, :, 1:] # [B, N, T-1]
                ade = loss[mask]
            else:
                loss = torch.sqrt(torch.sum((recon_flow - gt_flow) ** 2, dim=-1))[:, :, 1:] # [B, N, T-1]
                new_weight = torch.ones_like(loss)
                new_weight[loss > threshold] = weight
                ade = new_weight * loss
        else:
            ade = torch.sqrt(torch.sum((recon_flow - gt_flow) ** 2, dim=-1))  # [B, N, T-1]

        return ade, torch.mean(ade)
    
    def less_than_delta(self, ade, delta_list=[2, 4, 8]):
        ''' calculate the less than delta ratio
        '''
        proportions = []
        for threshold in delta_list:
            proportion = (ade <= threshold).float().mean()
            proportions.append(proportion)

        average_proportion = torch.mean(torch.tensor(proportions))

        return average_proportion

def ScaleDirectionCVAE_S(**kwargs):
    return ScaleDirectionCVAE(hidden_size=384, vae_encoder_layers=4, vae_decoder_layers=8, num_heads=4, **kwargs)

def ScaleDirectionCVAE_B(**kwargs):
    return ScaleDirectionCVAE(hidden_size=768, vae_encoder_layers=4, vae_decoder_layers=10, num_heads=4, **kwargs)

def ScaleDirectionCVAE_L(**kwargs):
    return ScaleDirectionCVAE(hidden_size=768, vae_encoder_layers=6, vae_decoder_layers=12, num_heads=8, **kwargs)

CVAE_models = {
    'CVAE-L':    ScaleDirectionCVAE_L,
    'CVAE-B':    ScaleDirectionCVAE_B,
    'CVAE-S':    ScaleDirectionCVAE_S,
}