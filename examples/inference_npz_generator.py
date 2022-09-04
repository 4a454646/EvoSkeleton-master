import sys
sys.path.append("../")

import libs.model.model as libm
from libs.dataset.h36m.data_utils import unNormalizeData
import cv2
import torch
import numpy as np
import imageio
import os

num_joints = 16
gt_3d = False
re_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
# paths
model_path = './example_model.th'
stats = np.load('./stats.npy', allow_pickle=True).item()
dim_used_2d = stats['dim_use_2d']
mean_2d = stats['mean_2d']
std_2d = stats['std_2d']
# load the checkpoint and statistics
ckpt = torch.load(model_path)
# print(data_dic)
# initialize the model
cascade = libm.get_cascade()
input_size = 32
output_size = 48
for stage_id in range(2):
    # initialize a single deep learner
    stage_model = libm.get_model(stage_id + 1,
                                 refine_3d=False,
                                 norm_twoD=False,
                                 num_blocks=2,
                                 input_size=input_size,
                                 output_size=output_size,
                                 linear_size=1024,
                                 dropout=0.5,
                                 leaky=False)
    cascade.append(stage_model)
cascade.load_state_dict(ckpt)
cascade.eval()


def normalize(skeleton, re_order=None):
    norm_skel = skeleton.copy()
    if re_order is not None:
        norm_skel = norm_skel[re_order].reshape(32)
    norm_skel = norm_skel.reshape(16, 2)
    mean_x = np.mean(norm_skel[:, 0])
    std_x = np.std(norm_skel[:, 0])
    mean_y = np.mean(norm_skel[:, 1])
    std_y = np.std(norm_skel[:, 1])
    denominator = (0.5 * (std_x + std_y))
    norm_skel[:, 0] = (norm_skel[:, 0] - mean_x) / denominator
    norm_skel[:, 1] = (norm_skel[:, 1] - mean_y) / denominator
    norm_skel = norm_skel.reshape(32)
    return norm_skel


def get_pred(cascade, data):
    """
    Get prediction from a cascaded model
    """
    # forward pass to get prediction for the first stage
    num_stages = len(cascade)
    # for legacy code that does not have the num_blocks attribute
    for i in range(len(cascade)):
        cascade[i].num_blocks = len(cascade[i].res_blocks)
    prediction = cascade[0](data)
    # prediction for later stages
    for stage_idx in range(1, num_stages):
        prediction += cascade[stage_idx](data)
    return prediction


def re_order(skeleton):
    skeleton = skeleton.copy().reshape(-1, 3)
    # permute the order of x,y,z axis
    skeleton[:, [0, 1, 2]] = skeleton[:, [0, 2, 1]]
    skeleton = skeleton.reshape(96)
    return skeleton


# load the compressed file keypoints.npz into jeff_keypoints
# two_d_keypoints_dir = "/home/jeff/Documents/Code/EvoSkeleton-master/examples/keypoints_npz/"
# for f in os.listdir(two_d_keypoints_dir):
# print(f"processing {two_d_keypoints_dir}{f}")
# l = np.load(two_d_keypoints_dir + f, allow_pickle=True)
l = np.load("chris_buff_kpts.npz", allow_pickle=True)
cur_2d_keypoints = l["reconstruction"][0]
# f = os.listdir("jeff_kpts_normalized_clipped")
for count in range(len(cur_2d_keypoints)):
    to_save = []
    skeleton_2d = cur_2d_keypoints[count]
    norm_ske_gt = normalize(skeleton_2d, re_order_indices).reshape(1, -1)
    pred = get_pred(cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))
    pred = unNormalizeData(
        pred.data.numpy(),
        stats['mean_3d'],
        stats['std_3d'],
        stats['dim_ignore_3d']
    )

    channels = re_order(pred)
    vals = np.reshape(channels, (32, -1))
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
    for i in np.arange(len(I)):
        if (i > 5):
            x, y, z = [np.array([vals[I[i], j]]) for j in range(3)]
            to_save.append([x[0], y[0], z[0]])
    # save the keypoints to a npz file
    #   z(f"/home/jeff/Documents/Code/EvoSkeleton-master/examples/keypoints_output/{f[:-4]}_3D_{count}.npz", to_save)
        np.savez(f"chris_buff_3d_kpts/chris_buff_3d_{count}.npz", to_save)
