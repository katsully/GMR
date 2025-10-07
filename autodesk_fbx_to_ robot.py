
import subprocess


fbx_path="./data/Z_Solo_Best.fbx"
inter_path = "./data/Z_Solo_Best_robot_inter.pkl"
output_path="./data/Z_Solo_Best_robot.pkl"
root_joint="Hips"
fps=60
robot="unitree_g1"

# Step 1: Process FBX file (run this with fbx_sdk environment)
print("Step 1: Converting FBX to intermediate format...")
subprocess.run(f"conda activate fbx_sdk && python third_party/poselib/fbx_importer.py --input {fbx_path} --output {inter_path} --root-joint {root_joint} --fps {fps}", shell=True, check=True)

# Step 2: Convert to robot motion (run this with gmr environment)
print("\nStep 2: Converting to robot motion...")
subprocess.run(f"conda activate g1_env && python scripts/fbx_offline_to_robot.py --motion_file {inter_path} --robot {robot} --save_path {output_path} --rate_limit", shell=True, check=True)




    