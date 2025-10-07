
import subprocess

file_name = "Z_Solo_Best"
fbx_path=f"./data/{file_name}.fbx"
inter_path = f"./data/{file_name}_robot_inter.pkl"
output_path=f"./data/{file_name}_robot.pkl"
root_joint="Hips"
fps=30
robot="unitree_g1"
video_output_path = f"videos/{file_name}_robot.mp4"
max_video_duration = 10.0  # seconds

CONVERT_FBX = True

if CONVERT_FBX:
    # Step 1: Process FBX file (run this with fbx_sdk environment)
    print("Step 1: Converting FBX to intermediate format...")
    subprocess.run(f"conda activate fbx_sdk && python third_party/poselib/fbx_importer.py --input {fbx_path} --output {inter_path} --root-joint {root_joint} --fps {fps}", shell=True, check=True)

    # Step 2: Convert to robot motion (run this with gmr environment)
    print("\nStep 2: Converting to robot motion...")
    subprocess.run(f"conda activate g1_env && python scripts/fbx_offline_to_robot.py --motion_file {inter_path} --robot {robot} --save_path {output_path} --rate_limit", shell=True, check=True)


print("\nStep 3: Visualising robot motion...")
subprocess.run(f"conda activate g1_env && python scripts/vis_robot_motion.py --robot {robot} --robot_motion_path {output_path} --record_video --video_path {video_output_path} --max_duration {max_video_duration}", shell=True, check=True)
    