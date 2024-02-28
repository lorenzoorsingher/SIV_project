import numpy as np
from common import rotationMatrixToEulerAngles


def compute_relative_pose_error(estimated_poses, ground_truth_poses):
    assert len(estimated_poses) == len(
        ground_truth_poses
    )  # The number of estimated and ground truth poses must be the same

    errors = []
    for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses):
        # Aligns estimated pose with ground truth pose
        aligned_est_pose = horn_alignment(est_pose, gt_pose)

        # Computes error between aligned poses
        translation_error = compute_translation_error(
            aligned_est_pose.translation, gt_pose.translation
        )
        rotation_error = compute_rotation_error(
            aligned_est_pose.rotation, gt_pose.rotation
        )

        # Aggregates error
        total_error = translation_error + rotation_error
        errors.append(total_error)

    # Computes aggregate error metric (RMSE??)
    aggregate_error = compute_aggregate_error(errors)

    return aggregate_error


def compute_relative_pose_error_custom(estimated_poses, ground_truth_poses):

    assert len(estimated_poses) == len(
        ground_truth_poses
    )  # The number of estimated and ground truth poses must be the same

    errors = []
    for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses):

        # Aligns estimated pose with ground truth pose
        aligned_est_pose = horn_alignment(est_pose, gt_pose)

        est_t = aligned_est_pose.T[-1]
        est_R = aligned_est_pose.T[:3].T

        gt_t = gt_pose.T[-1]
        gt_R = gt_pose.T[:3].T
        print(
            (rotationMatrixToEulerAngles(gt_R)[1] * 180 / np.pi).round(2),
            " ",
            (rotationMatrixToEulerAngles(est_pose.T[:3].T)[1] * 180 / np.pi).round(2),
            " ",
            (rotationMatrixToEulerAngles(est_R)[1] * 180 / np.pi).round(2),
        )
        # Computes error between aligned poses
        translation_error = compute_translation_error(est_t, gt_t)
        rotation_error = compute_rotation_error(est_R, gt_R)

        # Aggregates error
        total_error = translation_error  # + rotation_error
        errors.append(total_error)

    # Computes aggregate error metric (RMSE??)
    aggregate_error = compute_aggregate_error(errors)

    return aggregate_error


# Alignment of poses with Horn's method
def horn_alignment(estimated_poses, ground_truth_poses):
    # Computes centroids (centers of mass of both sets of poses)
    est_centroid = np.mean(estimated_poses, axis=0)
    gt_centroid = np.mean(ground_truth_poses, axis=0)

    # Centers the poses (Both sets of poses are translated so that their centroids coincide)
    centered_est_poses = estimated_poses - est_centroid
    centered_gt_poses = ground_truth_poses - gt_centroid

    # Computes covariance matrix
    covariance_matrix = np.dot(centered_est_poses.T, centered_gt_poses)

    # Performs SVD (Singular Value Decomposition (SVD) is performed on the covariance matrix to obtain the optimal rotation matrix)
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Computes optimal rotation (we extract the optimal rotation matrix from the matrices obtained through SVD)
    R = np.dot(Vt.T, U.T)

    # Applies rotation (to the aligned poses)
    aligned_est_poses = np.dot(centered_est_poses, R.T)

    # Computes translation offset (between the aligned poses)
    t = gt_centroid - np.dot(R, est_centroid)

    # Applies translation (to align the centroids of the aligned poses)
    aligned_est_poses += t

    return aligned_est_poses


def compute_translation_error(
    estimated_translation, ground_truth_translation
):  # TODO: ADD TO IN PARAMETERS TO compute_relative_pose_error
    """
    Computes the Euclidean distance between estimated and ground truth translation vectors.

    Parameters:
        estimated_translation (numpy array): Estimated translation vector.
        ground_truth_translation (numpy array): Ground truth translation vector.

    Returns:
        error (float): Euclidean distance between estimated and ground truth translations.
    """
    # Compute Euclidean distance between translations (estimated and ground truth)
    error = np.linalg.norm(estimated_translation - ground_truth_translation)
    return error


def compute_rotation_error(estimated_rotation, ground_truth_rotation):
    """
    Computes the rotation error between estimated and ground truth rotation matrices (angle difference between rotations)

    Parameters:
        estimated_rotation (numpy array): Estimated rotation matrix (3x3).
        ground_truth_rotation (numpy array): Ground truth rotation matrix (3x3).

    Returns:
        error (float): Angle difference between estimated and ground truth rotations (in radians).
    """
    # Computes the rotation matrix representing the rotation from estimated to ground truth
    delta_rotation = np.matmul(estimated_rotation.T, ground_truth_rotation)

    # The rotation angle is extracted from the rotation matrix
    # Axis-angle representation: the angle is the magnitude of the rotation
    # and the axis is the unit eigenvector (remains in the same direction after a linear transformation is applied to it) corresponding to the eigenvalue (scaling factor) 1 of the rotation matrix
    # The rotation matrices are assumed to be orthonormal (i.e., proper rotations)
    cos_theta = (np.trace(delta_rotation) - 1) / 2
    cos_theta = min(
        1, max(cos_theta, -1)
    )  # Ensure cos_theta is within [-1, 1] to avoid numerical errors
    angle = np.arccos(cos_theta)

    # angle_degrees = np.degrees(angle)

    return angle


def compute_aggregate_error(errors):
    """
    Computes the aggregate error metric (in this case RMSE) based on a list of individual errors.

    Parameters:
        errors (list): List of individual error values.

    Returns:
        aggregate_error (float): Aggregate error metric.
    """
    # Convert errors to numpy array for easy computation
    errors_array = np.array(errors)

    # Compute aggregate error metric (Root Mean Square Error - RMSE)
    aggregate_error = np.sqrt(np.mean(errors_array**2))

    return aggregate_error
