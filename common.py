import numpy as np

ORB_BF = 0
ORB_FLANN = 1
SIFT_FLANN = 2
SIFT_KNN = 3

# consts for FLANN
FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_KDTREE_SINGLE = 4
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_INDEX_SAVED = 254
FLANN_INDEX_AUTOTUNED = 255

IMG_DEBUG = True


def eval_error(old_gt_pose, gt_pose, old_agent_pose, agent_pose):

    gt_diff = (old_gt_pose - gt_pose)[:, 3]
    agent_diff = (old_agent_pose - agent_pose)[:, 3]

    return np.linalg.norm(gt_diff - agent_diff)
