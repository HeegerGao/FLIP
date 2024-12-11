import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from utils.network_utils import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed, HWPatchEmbed, FinalLayer, TimestepEmbedder, DiTCrossAttentionandAdaLNZero, SentenceEncoder
from einops import rearrange

class DynamicsModel(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=[32, 32],
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        flow_horizon=16,
        obs_history=4,
    ):
        super().__init__()
        assert depth % 2 == 0, "Depth must be divisible by 2."

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.flow_horizon = flow_horizon
        self.obs_history = obs_history
        self.input_size = input_size
        
        self.language_encoder = SentenceEncoder(
            hidden_size=4096,
            output_size=hidden_size,
            num_layers=3,
            num_heads=8,
        )

        if input_size[0] == input_size[1]:
            self.x_embedder = PatchEmbed(input_size[0], patch_size, in_channels, hidden_size, bias=True)
        else:
            self.x_embedder = HWPatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.flow_embedder = nn.Sequential(
            nn.Linear(2*self.flow_horizon, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding for spatial patches:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # Will use fixed sin-cos embedding for temporal patches:
        self.temp_embed = nn.Parameter(torch.zeros(1, self.obs_history + self.flow_horizon - 1, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTCrossAttentionandAdaLNZero(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        num_patches_h, num_patches_w = self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size
        img_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (num_patches_h, num_patches_w))
        img_embed = rearrange(img_embed, 'n d -> () n d')
        self.pos_embed.data.copy_(torch.from_numpy(img_embed))
        temp_embed = get_1d_sincos_pos_embed(self.temp_embed.shape[-1], self.obs_history + self.flow_horizon - 1)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        q = self.x_embedder.patch_size[1]
        h = self.input_size[0] // p
        w = self.input_size[1] // q
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, q, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * q))
        return imgs

    def forward(self, x, t, flow, sentence):
        """
        Forward pass of DynamicsModel.
        x: (B, T, C, H, W) tensor of videos. Note here T = obs_history + flow_horizon - 1
        t: (B,) tensor of diffusion timesteps
        flow: (B, N, T, 2) tensor of flows. Note here T = flow_horizon
        sentence: (B,) tensor of languages
        """

        B, T, C, H, W = x.shape

        sentence_embedding = self.language_encoder(sentence)  # (B, hidden_size)
        sentence_embedding_spatial = sentence_embedding.unsqueeze(1).repeat(1, self.flow_horizon+self.obs_history-1, 1).reshape(-1, sentence_embedding.shape[-1]) # expand to num steps
        sentence_embedding_temporal = sentence_embedding.unsqueeze(1).repeat(1, self.pos_embed.shape[1], 1).reshape(-1, sentence_embedding.shape[-1])  # expand to num_patches

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.x_embedder(x) + self.pos_embed  # (B*T, N, D), where N = H * W / patch_size ** 2
        x = rearrange(x, '(b t) n d -> (b n) t d', b=B)
        x += self.temp_embed
        x = rearrange(x, '(b n) t d -> (b t) n d', b=B)

        t = self.t_embedder(t)                   # (N, D)
        t_spatial = t.unsqueeze(1).repeat(1, self.flow_horizon+self.obs_history-1, 1).reshape(-1, t.shape[-1]) # expand to num steps
        t_temporal = t.unsqueeze(1).repeat(1, self.pos_embed.shape[1], 1).reshape(-1, t.shape[-1])  # expand to num_patches

        flow_embedding = self.flow_embedder(rearrange(flow, 'b n t d -> b n (t d)'))   # (B, N, D)
        _, N, D = flow_embedding.shape
        flow_embedding_spatial = flow_embedding.unsqueeze(1).repeat(1, self.flow_horizon+self.obs_history-1, 1, 1).reshape(-1, N, D)  # expand to num steps
        flow_embedding_temporal = flow_embedding.unsqueeze(1).repeat(1, self.pos_embed.shape[1], 1, 1).reshape(-1, N, D)  # expand to num_patches

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            c = t_spatial + sentence_embedding_spatial
            x = spatial_block(x, c, flow_embedding_spatial)

            x = rearrange(x, '(b t) n d -> (b n) t d', b=B)
            c = t_temporal + sentence_embedding_temporal
            x = temp_block(x, c, flow_embedding_temporal)
            x = rearrange(x, '(b n) t d-> (b t) n d', b=B)
        
        c = t_spatial + sentence_embedding_spatial
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B)

        return x


#################################################################################
#                       Dynamics Model Configs                                  #
#################################################################################

def DynamicsModel_L(**kwargs):
    return DynamicsModel(depth=16, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DynamicsModel_B(**kwargs):
    return DynamicsModel(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DynamicsModel_S(**kwargs):
    return DynamicsModel(depth=8, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

DynamicsModel_models = {
    'DynamicsModel-L':  DynamicsModel_L,
    'DynamicsModel-B':  DynamicsModel_B,
    'DynamicsModel-S':  DynamicsModel_S,
}