import numpy as np

ORB_BF = 0
ORB_FLANN = 1
SIFT_FLANN = 2
SIFT_KNN = 3

IMG_DEBUG = False


def eval_error(old_gt_pose, gt_pose, old_agent_pose, agent_pose):

    gt_diff = (old_gt_pose - gt_pose)[:, 3]
    agent_diff = (old_agent_pose - agent_pose)[:, 3]

    return np.linalg.norm(gt_diff - agent_diff)
