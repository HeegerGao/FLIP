from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from moviepy.editor import ImageSequenceClip
from moviepy.editor import concatenate_videoclips

def vis_pred_flow(init_img, flow):
    """
    Visualize the predicted flow on the dark initial image.

    Args:
        init_img: initial image, [3, H, W]
        flow: predicted flow, [1, N, T, 2]
    """
    init_img = init_img.clone().permute(1,2,0).detach().cpu().numpy()
    predicted_flow = flow[0].clone().detach().cpu().numpy()

    color_map = cm.get_cmap("gist_rainbow")
    N = predicted_flow.shape[0]
    T = predicted_flow.shape[1]
    
    # using the first frame to specify color
    vector_colors = np.zeros((N, T, 3)) # per point color. The point in the same row is the same color. 3 means RGB
    y_min, y_max = (
        predicted_flow[:, 0, 1].min(),
        predicted_flow[:, 0, 1].max(),
    )
    norm = plt.Normalize(y_min, y_max)
    for n in range(N):
        color = color_map(norm(predicted_flow[n, 0, 1]))
        color = np.array(color[:3])[None] * 255
        vector_colors[n, :] = np.repeat(color, T, axis=0)

    # res video
    res_video = []
    for _ in range(T):
        # MAKE VIDEO BLACK / alpha less
        a_channel = np.ones(init_img.shape, dtype=np.float64) / 2.0
        res_video.append(init_img * a_channel)
        # a_channel = np.ones(init_img.shape, dtype=np.float64)
        # res_video.append(a_channel)


    linewidth = 1

    for t in range(T):
        if t > 0:
            res_video[t] = res_video[t-1].copy()
        for i in range(N):
            coord = (int(predicted_flow[i, t, 0]), int(predicted_flow[i, t, 1]))
            if t > 0:
                coord_1 = (int(predicted_flow[i, t-1, 0]), int(predicted_flow[i, t-1, 1]))

            if coord[0] != 0 and coord[1] != 0:
                cv2.circle(res_video[t], coord, int(linewidth * 1), vector_colors[i, t].tolist(), thickness=-1 -1,)
                if t > 0:
                    cv2.line(res_video[t], coord_1, coord, vector_colors[i, t].tolist(), thickness=int(linewidth * 1))

    return res_video

def connect_and_save_videos(flow_video_clips, original_video, output_path, drop_last=True, T_video=None, fps_flow=10, fps_video=24, permute=True):
    '''
    args:
        flow_video_clips: list of N flow video list of flow_horizon frames [H, W, 3]
        original_video: original video tensor [T, 3, H, W]
        output_path: str, path to save the output video
    '''

    # get the number of clips
    N = len(flow_video_clips)
    T = len(flow_video_clips[0])
    if T_video is None:
        T_video = T

    video_list = []

    if drop_last:    # TODO: currently we omit the last incomplete clip
        total = N-1
    else:
        total = N
    for i in range(total):
        flow_v = ImageSequenceClip([flow_video_clips[i][t] for t in range(T)], fps=fps_flow)
        if permute:
            original_v = ImageSequenceClip([original_video[t].permute(1, 2, 0).cpu().numpy() for t in range(i*T_video, (i+1)*T_video)], fps=fps_video)
        else:
            original_v = ImageSequenceClip([original_video[t].cpu().numpy() for t in range(i*T_video, (i+1)*T_video)], fps=fps_video)

        video_list.append(flow_v)
        video_list.append(original_v)

    final_video = concatenate_videoclips(video_list)

    final_video.write_videofile(output_path, codec="libx264")
    
def connect_and_save_videos_different_length(flow_video_clips, original_video_clips, output_path, drop_last=True, fps_flow=10, fps_video=24, permute=True, T_videos=[]):
    '''
    args:
        flow_video_clips: list of N flow video list of flow_horizon frames [H, W, 3]
        original_video: original video tensor [T, 3, H, W]
        output_path: str, path to save the output video
    '''

    # get the number of clips
    N = len(flow_video_clips)
    T = len(flow_video_clips[0])

    video_list = []

    if drop_last:    # TODO: currently we omit the last incomplete clip
        total = N-1
    else:
        total = N
    for i in range(total):
        flow_v = ImageSequenceClip([flow_video_clips[i][t] for t in range(T)], fps=fps_flow)
        original_video = original_video_clips[i]
        if permute:
            original_v = ImageSequenceClip([original_video[t].permute(1, 2, 0).cpu().numpy() for t in range(len(original_video))], fps=fps_video)
        else:
            original_v = ImageSequenceClip([original_video[t].cpu().numpy() for t in range(len(original_video))], fps=fps_video)

        video_list.append(flow_v)
        video_list.append(original_v)

    final_video = concatenate_videoclips(video_list)

    final_video.write_videofile(output_path, codec="libx264")