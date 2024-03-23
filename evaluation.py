import json
import numpy as np
import matplotlib.pyplot as plt

# from scipy.interpolate import interp1d

from common import rotationMatrixToEulerAngles


def compute_error(est_pose, gt_pose, old_est_pose, old_gt_pose):
    """
    Computes the error between the estimated and ground truth poses using
    the method presented in the report.

    Parameters
    ----------
    - est_pose (ndarray): Estimated pose.
    - gt_pose (ndarray): Ground truth pose.
    - old_est_pose (ndarray): Previous estimated pose.
    - old_gt_pose (ndarray): Previous ground truth pose.

    Returns
    -------
    - float: The error between the estimated and ground truth poses.
    """

    delta_t_est = est_pose[:, -1] - old_est_pose[:, -1]  # distanza percorsa stimata
    delta_t_gt = gt_pose[:, -1] - old_gt_pose[:, -1]  # distanza percorsa ground truth

    delta_R_est = np.matmul(
        est_pose[:3, :3], old_est_pose[:3, :3].T
    )  # estimated rotation
    delta_R_gt = gt_pose[:3, :3] @ old_gt_pose[:3, :3].T  # GT rotation

    err_rot = delta_R_est @ delta_R_gt.T
    err_rot_deg = abs(rotationMatrixToEulerAngles(err_rot)[1] * 180 / np.pi)

    diff_rot = gt_pose[:3, :3] @ est_pose[:3, :3].T

    delta_t_est_rot = diff_rot @ delta_t_est

    err_trasl = np.linalg.norm(
        delta_t_gt - delta_t_est_rot
    )  # in questo caso stiamo prendendo in considerazione il vettore distanza stimato allineato nella direzione del vettore transl. gt

    err_trasl_no_align = np.linalg.norm(
        delta_t_gt - delta_t_est
    )  # proviamo a fare la differenza senza riallineamento, perché sono solo vettori distanza, quindi rappresentano la distanza percorsa, che dovrebbe essere relativa e non globale, hence no need for alignment

    total_err = err_rot_deg + err_trasl
    total_err = total_err / (1 / 2 + total_err)

    return total_err


def save_metrics(
    est_poses, gt_poses, errors, settings, output_path="data/output", steps_sec=0
):
    """
    Saves metrics (estimated poses, ground truth poses, and errors) to a file.

    Parameters
    ----------
    - est_poses (list): List of estimated poses.
    - gt_poses (list): List of ground truth poses.
    - errors (list): List of errors.
    - settings (dict): Dictionary of settings.
    - output_path (str): Path to the output file.
    - steps_sec (int): Steps per second.
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

    settings["avg_error"] = np.mean(errors).round(3)
    settings["max_error"] = np.max(errors).round(3)
    settings["steps_sec"] = steps_sec

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

    fig, ax = plt.subplots()
    ax.plot(errors)

    ax.set_ylim([0, 1])  # Fix y-axis to 1
    plt.savefig(output_path + "/error.png", bbox_inches="tight")

    print("Metrics saved to", output_path)
    print("avg: ", np.mean(errors).round(3), " max: ", np.max(errors).round(3))


def save_metrics_video(est_poses, settings, output_path="data/output", steps_sec=0):
    """
    Saves metrics (estimated poses, ground truth poses, and errors) to a file.

    Parameters
    ----------
    - est_poses (list): List of estimated poses.
    - settings (dict): Dictionary of settings.
    - output_path (str): Path to the output file.
    """
    est_dump = output_path + "/est.json"

    settings_path = output_path + "/settings.json"
    flattend = [np.array(pos).flatten().tolist() for pos in est_poses]
    with open(est_dump, "w", encoding="utf-8") as f:
        json.dump(flattend, f, ensure_ascii=False, indent=4)

    settings["steps_sec"] = steps_sec

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

    print("Metrics saved to", output_path)
