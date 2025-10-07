from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str,
                        default="videos/example.mp4")
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Maximum duration in seconds (stops after this time)")

    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    max_frames = None
    if args.max_duration is not None:
        max_frames = int(args.max_duration * motion_fps)

    frames_played = 0
    while True:
        env.step(motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True)
        frame_idx += 1
        frames_played += 1

        # Stop if max_duration reached
        if max_frames is not None and frames_played >= max_frames:
            break

        if frame_idx >= len(motion_root_pos):
            if args.record_video:
                break  # Stop after one loop when recording
            frame_idx = 0
    env.close()