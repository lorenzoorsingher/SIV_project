import json
import numpy as np
import matplotlib.pyplot as plt

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
    # error = np.log2(np.linalg.norm(estimated_pose - ground_truth_pose) + 1) -> scaling of error (logarithmically) for better representation (+1 for numerical stability)
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


def compute_mockup_error(estimated_pose, ground_truth_pose):
    """
    Mockup function to compute the error between estimated and ground truth poses.

    Parameters:
        estimated_pose (numpy array): Estimated pose.
        ground_truth_pose (numpy array): Ground truth pose.

    Returns:
        error (float): Error between estimated and ground truth poses.
    """
    # First, the poses are aligned
    aligned_est_pose = horn_alignment(estimated_pose, ground_truth_pose)

    # Then, the aligned estimated poses are used to compute the translation error
    translation_error = compute_translation_error(
        aligned_est_pose[:, -1], ground_truth_pose[:, -1]
    )

    # And the rotation error
    rotation_error = compute_rotation_error(
        aligned_est_pose[:, :3], ground_truth_pose[:, :3]
    )

    align_rot = (
        rotationMatrixToEulerAngles(aligned_est_pose[:, :3])[1] * 180 / np.pi
    ).round(2)
    gt_rot = (
        rotationMatrixToEulerAngles(ground_truth_pose[:, :3])[1] * 180 / np.pi
    ).round(2)
    est_rot = (
        rotationMatrixToEulerAngles(estimated_pose[:, :3])[1] * 180 / np.pi
    ).round(2)

    print(
        "al: ",
        align_rot,
        "\tgt: ",
        gt_rot,
        "\test: ",
        est_rot,
    )
    print(
        "translation error: ",
        translation_error.round(2),
        " rotation error: ",
        (rotation_error * 180 / np.pi).round(2),
    )
    total_error = translation_error + rotation_error

    # Compute error between estimated and ground truth poses
    # error = np.log2(np.linalg.norm(estimated_pose - ground_truth_pose) + 1)

    # return error
    return total_error


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_mockup_error2(est_pose, gt_pose, old_est_pose, old_gt_pose):

    # extraction of translation vectors -> commenting first two lines jsut to be sure of the sintax
    # delta_t_est = est_pose[:3, 3] - old_est_pose[:3, 3]  # estimated travelled distance
    # delta_t_gt = gt_pose[:3, 3] - old_gt_pose[:3, 3]  # ground truth travelled distance
    delta_t_est = est_pose[:, -1] - old_est_pose[:, -1]  # distanza percorsa stimata
    delta_t_gt = gt_pose[:, -1] - old_gt_pose[:, -1]  # distanza percorsa ground truth
    # okay no, it's the same thing, just a different sintax :)

    delta_R_est = np.matmul(
        est_pose[:3, :3], old_est_pose[:3, :3].T
    )  # estimated rotation
    delta_R_gt = gt_pose[:3, :3] @ old_gt_pose[:3, :3].T  # GT rotation

    # delta_R_est_deg = rotationMatrixToEulerAngles(delta_R_est)[1] * 180 / np.pi
    # delta_R_gt_deg = rotationMatrixToEulerAngles(delta_R_gt)[1] * 180 / np.pi

    err_rot = delta_R_est @ delta_R_gt.T
    err_rot_deg = rotationMatrixToEulerAngles(err_rot)[1] * 180 / np.pi
    err_rot_deg = sigmoid(abs(err_rot_deg)) - 0.5

    diff_rot = gt_pose[:3, :3] @ est_pose[:3, :3].T
    diff_rot_deg = rotationMatrixToEulerAngles(diff_rot)[1] * 180 / np.pi

    ### delta_t_est_rot = diff_rot @ delta_t_est # -> originale, ma ho il sospetto che diff_rot stoni logicamente
    delta_t_est_rot = (
        err_rot @ delta_t_est
    )  # in questo modo dovremmo star direzionando la camera del vettore direzione stimato nella dir della camera GT
    # print("#" * int(err_rot_deg * 10 + 1))
    # print("diff_rot:", diff_rot_deg.round(2))
    # err_trasl = sigmoid(abs(np.linalg.norm(delta_t_gt - delta_t_est_rot)) - 0.5) -> lei è quella originale

    err_trasl = np.linalg.norm(
        delta_t_gt - delta_t_est_rot
    )  # in questo caso stiamo prendendo in considerazione il vettore distanza stimato allineato nella direzione del vettore transl. gt
    err_trasl_noAlign = np.linalg.norm(
        delta_t_gt - delta_t_est
    )  # proviamo a fare la differenza senza riallineamento, perché sono solo vettori distanza, quindi rappresentano la distanza percorsa, che dovrebbe essere relativa e non globale, hence no need for alignment

    # print("translation error with alignement:", (err_trasl).round(4))
    # print("translation error:", (err_trasl_noAlign).round(4))

    print("ROTATION ERROR:", diff_rot_deg)
    print("difference among errors:", (err_trasl - err_trasl_noAlign).round(4))
    print("gt distance travelled:", (delta_t_gt).round(4))
    print("estimated distance travelled:", (delta_t_est).round(4))

    # err_trans_2 = np.linalg.norm(delta_t_gt - delta_t_est)
    # print(err_trans.round(2), " \t", "#" * int(err_trans * 10 + 1))
    # print(err_trans_2.round(2), " \t", "&" * int(err_trans_2 * 10 + 1))
    # print("")
    # breakpoint()
    # print(err_rot_deg.round(2) + err_trasl.round(2))

    total_erro = err_rot_deg + err_trasl
    # total_erro_NA = err_rot_deg + err_trasl_noAlign
    print("error with alignment:", err_trasl)
    print("error NO alignment:", err_trasl_noAlign)
    return total_erro


def save_metrics(est_poses, gt_poses, errors, settings, output_path="data/output"):
    """
    Saves metrics (estimated poses, ground truth poses, and errors) to a file.

    Parameters:
        est_poses (list): List of estimated poses.
        gt_poses (list): List of ground truth poses.
        errors (list): List of errors.
        output_path (str): Path to the output file.
    """
    est_dump = output_path + "/est.json"
    gt_dump = output_path + "/gt.json"
    err_dump = output_path + "/err.json"
    settings_path = output_path + "/settings.json"
    flattend = [np.array(pos).flatten().tolist() for pos in est_poses]
    with open(est_dump, "w", encoding="utf-8") as f:
        json.dump(flattend, f, ensure_ascii=False, indent=4)

    flattend = [pos.flatten().tolist() for pos in gt_poses]
    with open(gt_dump, "w", encoding="utf-8") as f:
        json.dump(flattend, f, ensure_ascii=False, indent=4)

    with open(err_dump, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=4)

    settings["avg_error"] = str(np.mean(errors).round(3))
    settings["max_error"] = str(np.max(errors).round(3))

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
    fig, ax = plt.subplots()
    ax.plot(errors)
    plt.savefig(output_path + "/error.png", bbox_inches="tight")

    print("Metrics saved to", output_path)
    print("avg: ", np.mean(errors).round(3), " max: ", np.max(errors).round(3))
