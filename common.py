import math
import numpy as np
import cv2


def _form_transf(R, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector

    Parameters
    ----------
    R (ndarray): The rotation matrix
    t (list): The translation vector

    Returns
    -------
    T (ndarray): The transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def decomp_essential_mat(E, q1, q2, mtx, initP):
    """
    Decompose the Essential matrix

    Parameters
    ----------
    E (ndarray): Essential matrix
    q1 (ndarray): The good keypoints matches position in i-1'th image
    q2 (ndarray): The good keypoints matches position in i'th image

    Returns
    -------
    right_pair (list): Contains the rotation matrix and translation vector
    """

    def sum_z_cal_relative_scale(R, t):
        # Get the transformation matrix
        T = _form_transf(R, t)
        # Make the projection matrix
        P = np.matmul(np.concatenate((mtx, np.zeros((3, 1))), axis=1), T)

        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(initP, P, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        # Find the number of points there has positive z coordinate in both cameras
        sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

        # Form point pairs and calculate the relative scale
        relative_scale = np.mean(
            np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
            / np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)
        )
        if math.isnan(relative_scale):
            relative_scale = 1
        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

    # Decompose the essential matrix
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)

    # Make a list of the different possible pairs
    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

    # Check which solution there is the right one
    z_sums = []
    relative_scales = []
    for R, t in pairs:
        z_sum, scale = sum_z_cal_relative_scale(R, t)
        z_sums.append(z_sum)
        relative_scales.append(scale)

    # Select the pair there has the most points with positive z coordinate
    right_pair_idx = np.argmax(z_sums)
    right_pair = pairs[right_pair_idx]
    relative_scale = relative_scales[right_pair_idx]
    R1, t = right_pair
    t = t * relative_scale

    return [R1, t]
