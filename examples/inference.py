import sys
sys.path.append("../")

import libs.model.model as libm
from libs.dataset.h36m.data_utils import unNormalizeData
import cv2
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

num_joints = 16
gt_3d = False
# pose_connection = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
#                    [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
pose_connection = [[0, 7 - 6], [7 - 6, 8 - 6], [8 - 6, 9 - 6], [9 - 6, 10 - 6], [8 - 6, 11 - 6], [11 - 6, 12 - 6], [12 - 6, 13 - 6], [8 - 6, 14 - 6], [14 - 6, 15 - 6], [15 - 6, 16 - 6]]
# realign keypoints after stuff was made missing
# 16 out of 17 key-points are used as inputs in this examplar model
re_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
# paths
data_dic_path = './example_annot.npy'
model_path = './example_model.th'
stats = np.load('./stats.npy', allow_pickle=True).item()
dim_used_2d = stats['dim_use_2d']
mean_2d = stats['mean_2d']
std_2d = stats['std_2d']
# load the checkpoint and statistics
ckpt = torch.load(model_path)
data_dic = np.load(data_dic_path, allow_pickle=True).item()
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
# process and show total_to_show examples
count = 0
total_to_show = 10


def draw_skeleton(ax, skeleton, gt=False, add_index=True):
    for segment_idx in range(len(pose_connection)):
        point1_idx = pose_connection[segment_idx][0]
        point2_idx = pose_connection[segment_idx][1]
        point1 = skeleton[point1_idx]
        point2 = skeleton[point2_idx]
        color = 'k' if gt else 'r'
        plt.plot([int(point1[0]), int(point2[0])],
                 [int(point1[1]), int(point2[1])],
                 c=color,
                 linewidth=2)
    # if add_index:
    #     for (idx, re_order_idx) in enumerate(re_order_indices):
    #         plt.text(skeleton[re_order_idx][0],
    #                  skeleton[re_order_idx][1],
    #                  str(idx + 1),
    #                  color='b'
    #                  )
    return


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


def show3Dpose(channels,
               ax,
               lcolor="#3498db",
               rcolor="#e74c3c",
               gt=False,
               pred=False
               ):
    vals = np.reshape(channels, (32, -1))
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
    # Make connection matrix
    # print(vals)
    outs = []
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        x = ((x/500)+1)/2
        y = ((y/500)+1)/2
        z = ((z/500)+1)/2
        if (i > 5):
            # print(x,y,z)
            ax.plot(x, y, z, lw=7)
            ax.scatter(x, y, z, s=200)
            x, y, z = [vals[I[i], j] for j in range(3)]
            outs.append([((x/500)+1)/2,((y/500)+1)/2,((z/500)+1)/2])
    # Get rid of the panes (actually, make them white)
    ax.set_xlim3d([0, 1])
    ax.set_ylim3d([0, 1])
    ax.set_zlim3d([0, 0.75])
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
    ax.invert_zaxis()
    return outs


def re_order(skeleton):
    skeleton = skeleton.copy().reshape(-1, 3)
    # permute the order of x,y,z axis
    skeleton[:, [0, 1, 2]] = skeleton[:, [0, 2, 1]]
    skeleton = skeleton.reshape(96)
    # print(skeleton)
    # print()
    # skeleton = skeleton[12:]
    # print(skeleton)
    return skeleton


def plot_3d_ax(ax,
               elev,
               azim,
               pred,
               title=None
               ):
    ax.view_init(elev=elev, azim=azim)
    return show3Dpose(re_order(pred), ax)


def adjust_figure(left=0,
                  right=1,
                  bottom=0.01,
                  top=0.95,
                  wspace=0,
                  hspace=0.4
                  ):
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    return


# use opencv to read video at /home/jeff/Documents/Code/StridedTransformer-Pose3D/demo/video/jeff_super_short.mp4, and put each frame into a list called video_frames
video_frames = []
cap = cv2.VideoCapture('isaac_good_scaled.mp4')
# cap = cv2.VideoCapture('ian_test_0_scaled.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        video_frames.append(frame)
    else:
        break
cap.release()

# load the compressed file keypoints.npz into jeff_keypoints
test_model = tf.keras.models.load_model('prettygoodtest.h5')
val_model = tf.keras.models.load_model('prettygoodval.h5')
l = np.load('jeff_test2_kpts.npz', allow_pickle=True)
jeff_keypoints = l["reconstruction"][0]
# for i in range(len(jeff_keypoints)):
f = plt.figure(figsize=(15, 15))
for i in tqdm(range(200, 300)):
    # img_path = './imgs/2.jpg'
    # img = imageio.imread(img_path)
    
    # ax1 = plt.subplot(131)
    # ax1.imshow(img)
    # plt.title('Input image')
    skeleton_2d = jeff_keypoints[i]
    # Nose was not used for this examplar model
    norm_ske_gt = normalize(skeleton_2d, re_order_indices).reshape(1, -1)
    pred = get_pred(cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))
    pred = unNormalizeData(pred.data.numpy(),
                           stats['mean_3d'],
                           stats['std_3d'],
                           stats['dim_ignore_3d']
                           )
    # pred = np.concatenate(([pred[0]], pred[7:]))
    ax3 = plt.subplot(1, 1, 1, projection='3d')
    # pred = np.array(pred[0][:12])
    outs = plot_3d_ax(ax=ax3, pred=pred, elev=5, azim=-80)
    # print(outs)
    plt.savefig(f'outs/{i}_pred.png')
    img = cv2.imread(f'outs/{i}_pred.png')
    proper = np.expand_dims(np.array(outs), axis=0)
    # print(proper.shape)
    pose_pred = val_model.predict(proper)[0]
    final = ""
    if (pose_pred[0] >= pose_pred[1]):
        final = "good"
    else:
        final = "bad"
    cv2.putText(img, final, (img.shape[1] // 2, img.shape[0]*5//6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(f'outs/{i}_pred.png', img)
    # print(i)
    ax3.clear()
