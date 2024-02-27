import numpy as np


def compute_relative_pose_error(estimated_poses, ground_truth_poses):
    assert len(estimated_poses) == len(
        ground_truth_poses
    )  # The number of estimated and ground truth poses must be the same

    errors = []
    for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses):
        # Align estimated pose with ground truth pose
        aligned_est_pose = horn_alignment(est_pose, gt_pose)

        # Compute error between aligned poses
        translation_error = compute_translation_error(
            aligned_est_pose.translation, gt_pose.translation
        )
        rotation_error = compute_rotation_error(
            aligned_est_pose.rotation, gt_pose.rotation
        )

        # Aggregate error
        total_error = translation_error + rotation_error
        errors.append(total_error)

    # Compute aggregate error metric (RMSE??)
    aggregate_error = compute_aggregate_error(errors)

    return aggregate_error


# Alignment of poses with Horn's method
def horn_alignment(estimated_poses, ground_truth_poses):
    # Compute centroids (centers of mass of both sets of poses)
    est_centroid = np.mean(estimated_poses, axis=0)
    gt_centroid = np.mean(ground_truth_poses, axis=0)

    # Center the poses (Both sets of poses are translated so that their centroids coincide)
    centered_est_poses = estimated_poses - est_centroid
    centered_gt_poses = ground_truth_poses - gt_centroid

    # Compute covariance matrix
    covariance_matrix = np.dot(centered_est_poses.T, centered_gt_poses)

    # Perform SVD (Singular Value Decomposition (SVD) is performed on the covariance matrix to obtain the optimal rotation matrix)
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Compute optimal rotation (we extract the optimal rotation matrix from the matrices obtained through SVD)
    R = np.dot(Vt.T, U.T)

    # Apply rotation (to the aligned poses)
    aligned_est_poses = np.dot(centered_est_poses, R.T)

    # Compute translation offset (between the aligned poses)
    t = gt_centroid - np.dot(R, est_centroid)

    # Apply translation (to align the centroids of the aligned poses)
    aligned_est_poses += t

    return aligned_est_poses


def compute_translation_error(estimated_translation, ground_truth_translation):
    """
    Compute the Euclidean distance between estimated and ground truth translation vectors.

    Parameters:
        estimated_translation (numpy array): Estimated translation vector.
        ground_truth_translation (numpy array): Ground truth translation vector.

    Returns:
        error (float): Euclidean distance between estimated and ground truth translations.
    """
    # Compute Euclidean distance between translations (estimated and ground truth)
    error = np.linalg.norm(estimated_translation - ground_truth_translation)
    return error
