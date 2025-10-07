import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import json, os


# pkl to npz modified from csv to npz from mocap converter

# -----------------------------------------------
# 1. Quaternion Helper Functions
# -----------------------------------------------
def quaternion_inverse(q):
    """Input q: (w, x, y, z), returns its inverse."""
    w, x, y, z = q
    norm_sq = w*w + x*x + y*y + z*z
    if norm_sq < 1e-8:
        norm_sq = 1e-8
    return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq

def quaternion_multiply(q1, q2):
    """Input/output: (w, x, y, z)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=q1.dtype)

def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    """
    Compute angular velocity from adjacent quaternions (w, x, y, z):
      - Relative rotation q_rel = inv(q_prev) * q_next
      - Extract rotation angle and axis from q_rel
      - Return (angle / dt) * axis
    """
    q_inv = quaternion_inverse(q_prev)
    q_rel = quaternion_multiply(q_inv, q_next)
    norm_q_rel = np.linalg.norm(q_rel)
    if norm_q_rel < eps:
        return np.zeros(3, dtype=np.float32)
    q_rel /= norm_q_rel

    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w*w)
    if sin_half < eps:
        return np.zeros(3, dtype=np.float32)
    axis = q_rel[1:] / sin_half
    return (angle / dt) * axis


# -----------------------------------------------
# 2. Helper Function to Build Pinocchio RobotWrapper
# -----------------------------------------------
def build_pin_robot(urdf_path, mesh_dir):
    """
    Load URDF file and construct a pin.RobotWrapper with free-flyer.
    Args:
        urdf_path: Path to the URDF file
        mesh_dir: Directory containing associated mesh files
    Returns:
        robot (pin.RobotWrapper)
    """
    # Note: If URDF already contains floating joint, you can modify this to use BuildFromURDF(urdf_path, ...)
    robot = RobotWrapper.BuildFromURDF(
        urdf_path,
        mesh_dir,
        pin.JointModelFreeFlyer()
    )
    return robot


# -----------------------------------------------
# 3. Main Conversion Pipeline
# -----------------------------------------------
def pkl_to_npz_main(pkl_path, u_path, m_dir,out_filename):
    
    with open(pkl_path,"rb") as f:
        data = pickle.load(f)
        
    if not isinstance(data, dict):
        raise(f"Skipping {pkl_path} (not a dict format)")
  
    # Skip if required keys are missing
    if "dof_pos" not in data:
        raise(f"Skipping {pkl_path} (missing required keys)")
     

    dof_pos = data["dof_pos"]
    frame_rate = data["fps"]    #FIXME why is fps 120 when fps input is 30?        
    motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
    motion[:, :3] = data["root_pos"]
    motion[:, 3:7] = data["root_rot"]
    motion[:, 7:] = dof_pos
    
    if frame_rate > 30:
        # downsample to 30 fps
        downsample_factor = frame_rate / 30.0
        indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
        old_length = motion.shape[0]
        motion = motion[indices]
        print(f"Downsampled from {old_length} to {motion.shape[0]} frames")
    # 3.1 Read CSV data and extract desired frame range (changed to frames 250~550)

    df = pd.DataFrame(motion)
    data_orig = df.to_numpy(dtype=np.float32)
    N_orig = data_orig.shape[0]


    # Original CSV: first 7 columns are root data, remaining are joint data
    root_data_orig = data_orig[:, :7]      # (N_orig, 7)
    joint_data_orig = data_orig[:, 7:]       # (N_orig, D)

    # XSens - need to start at second column because the first column is frame numbers
    # because in the config file we set the starting index to 1, we don't need to change the x value
    # root_data_orig = data_orig[:, 1:8]      # (N_orig, 7)
    # joint_data_orig = data_orig[:, 8:]       # (N_orig, D)

    # 3.2 Define original sampling rate (30fps) and new sampling rate (60fps), construct time series
    fps_orig = 30
    dt_orig = 1.0 / fps_orig
    t_orig = np.linspace(0, (N_orig - 1) * dt_orig, N_orig)

    fps_new = 30 #this used to be 60, not sure why it's this but it has other dependencies, but it's causing output to speed, changing to 30 to test
    dt_new = 1.0 / fps_new
    N_new = 2 * N_orig - 1   # Insert one new frame between every two frames
    t_new = np.linspace(0, (N_orig - 1) * dt_orig, N_new)

    # 3.3 Interpolate root_data positions and joint angles
    # Linear interpolation for positions (first three columns)
    root_pos_interp = interp1d(t_orig, root_data_orig[:, 0:3], axis=0, kind='linear')(t_new)

    # Slerp interpolation for quaternions (qx, qy, qz, qw)
    # Note: Quaternions in CSV are stored as (qx, qy, qz, qw), which matches scipy requirements
    rotations_orig = R.from_quat(root_data_orig[:, 3:7])
    slerp = Slerp(t_orig, rotations_orig)
    rotations_new = slerp(t_new)
    root_quat_interp = rotations_new.as_quat()  # (N_new, 4) still (qx, qy, qz, qw)

    # Combine interpolated root data
    root_data = np.hstack((root_pos_interp, root_quat_interp))  # (N_new, 7)

    # Linear interpolation for joint angles (joint_data)
    joint_data = interp1d(t_orig, joint_data_orig, axis=0, kind='linear')(t_new)

    # Update frame count, sampling rate, and time interval
    N = N_new
    fps = fps_new
    dt = dt_new

    # 3.4 Define joint names
    # this is for G1
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ]
    dof_names = np.array(joint_names, dtype=np.str_)

    # 3.5 Get joint positions (excluding Root)
    dof_positions = joint_data.copy()      # shape: (N, D)

    # 3.6 Calculate joint velocities (central differences + boundary forward/backward differences + Gaussian smoothing)
    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt
    dof_velocities_smoothed = gaussian_filter1d(dof_velocities, sigma=1, axis=0)

    # 3.7 Specify link names to record and get their poses in global coordinate frame
    # added names based on updates from perry
    body_names = [
        "pelvis", 
        # "head_link",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_elbow_link",
        "right_elbow_link",
        "right_hip_yaw_link",
        "left_hip_yaw_link",
        "right_rubber_hand",
        "left_rubber_hand",
        "right_ankle_roll_link",
        "left_ankle_roll_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "torso_link",
        "right_knee_link",
        "left_knee_link"
    ]

    body_names = np.array(body_names, dtype=np.str_)
    B = len(body_names)

    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    # 3.8 Build pin.RobotWrapper
    #    (Please change urdf_path and mesh_dir to your actual paths)
    urdf_path = u_path
    mesh_dir = m_dir
    robot = build_pin_robot(urdf_path, mesh_dir)
    model = robot.model
    data_pk = robot.data

    nq = model.nq  # Total DOF (including free-flyer)
    if (7 + joint_data.shape[1]) != nq:
        print(f"Warning: CSV columns={7 + joint_data.shape[1]}, but pinocchio nq={nq}, may need to check or adjust script parsing.")

    # 3.9 Perform forward kinematics (FK) for each frame to get link poses in world coordinate frame
    q_pin = pin.neutral(model)

    for i in range(N):
        # Set free-flyer
        q_pin[0:3] = root_data[i, 0:3]
        # Quaternions stored in root_data are in (qx, qy, qz, qw) order
        q_pin[3:7] = root_data[i, 3:7]
        # Other joints
        dofD = joint_data.shape[1]
        q_pin[7:7 + dofD] = joint_data[i, :]

        # Forward kinematics
        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)

        # Read and save global poses for each link
        for j, link_name in enumerate(body_names):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]  # Link transformation in world frame

            # Translation
            body_positions[i, j, :] = link_tf.translation
            # Rotation (pin.Quaternion defaults to (x,y,z,w), need to convert to (w,x,y,z))
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            body_rotations[i, j, :] = np.array([quat_xyzw.w,
                                                quat_xyzw.x,
                                                quat_xyzw.y,
                                                quat_xyzw.z],
                                               dtype=np.float32)

    # 3.10 Calculate body linear and angular velocities (in world coordinate frame)
    # -- Linear velocities: central differences --
    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt
    body_linear_velocities = gaussian_filter1d(body_linear_velocities, sigma=1, axis=0)

    # -- Angular velocities: computed from adjacent quaternions (in world coordinate frame) --
    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for j in range(B):
        quats = body_rotations[:, j, :]
        angular_vels = np.zeros((N, 3), dtype=np.float32)
        if N > 1:
            angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
            angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, N - 1):
            av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
            angular_vels[k] = 0.5 * (av1 + av2)
        # Smoothing
        body_angular_velocities[:, j, :] = gaussian_filter1d(angular_vels, sigma=1, axis=0)

    # 3.11 Package and save to NPZ
    data_dict = {
        "fps": fps,                                   # int64 scalar, sampling rate
        "dof_names": dof_names,                       # unicode array (D,)
        "body_names": body_names,                     # unicode array (B,)
        "dof_positions": dof_positions,               # float32 (N, D)
        "dof_velocities": dof_velocities_smoothed,    # float32 (N, D)
        "body_positions": body_positions,             # float32 (N, B, 3)
        "body_rotations": body_rotations,             # float32 (N, B, 4) (w,x,y,z)
        "body_linear_velocities": body_linear_velocities,     # float32 (N, B, 3)
        "body_angular_velocities": body_angular_velocities    # float32 (N, B, 3)
    }

    

    np.savez(out_filename, **data_dict)

    print(f"Conversion completed, data saved to {out_filename}")
    print("fps:", fps)
    print("dof_names:", dof_names.shape)
    print("body_names:", body_names.shape)
    print("dof_positions:", dof_positions.shape)
    print("dof_velocities:", dof_velocities_smoothed.shape)
    print("body_positions:", body_positions.shape)
    print("body_rotations:", body_rotations.shape)
    print("body_linear_velocities:", body_linear_velocities.shape)
    print("body_angular_velocities:", body_angular_velocities.shape)

