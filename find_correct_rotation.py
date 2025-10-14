import numpy as np
from scipy.spatial.transform import Rotation as R

# Current rotation offsets
old_leg_rotation = np.array([0.70710678, 0.0, 0.0, -0.70710678])  # Legs/hips/torso (scalar first)
left_elbow_rotation = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # Left elbow (working correctly)
right_elbow_rotation = np.array([-0.70710678, 0.70710678, 0.0, 0.0])  # Right elbow (working correctly)

print("=== CURRENT ROTATION OFFSETS ===")
print(f"Legs/Hips/Torso: {old_leg_rotation} (NOT working)")
print(f"Left Elbow/Wrist: {left_elbow_rotation} (working correctly)")
print(f"Right Elbow/Wrist: {right_elbow_rotation} (working correctly)")

# Convert to scipy Rotation objects (scalar_first=True matches the config file format)
r_leg = R.from_quat(old_leg_rotation, scalar_first=True)
r_left_elbow = R.from_quat(left_elbow_rotation, scalar_first=True)
r_right_elbow = R.from_quat(right_elbow_rotation, scalar_first=True)

# Calculate the difference transformation needed
# To go from leg rotation to elbow rotation:
# needed_transform * r_leg = r_elbow
# needed_transform = r_elbow * r_leg.inv()

transform_to_left = r_left_elbow * r_leg.inv()
transform_to_right = r_right_elbow * r_leg.inv()

print("\n=== TRANSFORMATION NEEDED ===")
print("To make legs look like left elbow/wrist:")
print(f"  Quaternion (w,x,y,z): {transform_to_left.as_quat(scalar_first=True)}")
print(f"  Euler angles (degrees): {transform_to_left.as_euler('xyz', degrees=True)}")

print("\nTo make legs look like right elbow/wrist:")
print(f"  Quaternion (w,x,y,z): {transform_to_right.as_quat(scalar_first=True)}")
print(f"  Euler angles (degrees): {transform_to_right.as_euler('xyz', degrees=True)}")

# Now let's calculate what the NEW rotation offset should be for legs
# New offset = transform * old_offset
new_left_leg = (transform_to_left * r_leg).as_quat(scalar_first=True)
new_right_leg = (transform_to_right * r_leg).as_quat(scalar_first=True)

print("\n=== NEW ROTATION OFFSETS FOR LEGS ===")
print(f"Left leg (to match left elbow style): {new_left_leg}")
print(f"Right leg (to match right elbow style): {new_right_leg}")

# Verify
print("\n=== VERIFICATION ===")
test_left = (R.from_quat(new_left_leg, scalar_first=True)).as_quat(scalar_first=True)
test_right = (R.from_quat(new_right_leg, scalar_first=True)).as_quat(scalar_first=True)
print(f"Left leg result: {test_left}")
print(f"Expected (left elbow): {left_elbow_rotation}")
print(f"Match: {np.allclose(test_left, left_elbow_rotation)}")

print(f"\nRight leg result: {test_right}")
print(f"Expected (right elbow): {right_elbow_rotation}")
print(f"Match: {np.allclose(test_right, right_elbow_rotation)}")

print("\n=== RECOMMENDATIONS ===")
print("Based on your image, all the 'other' joints are oriented the same way (wrong way).")
print("The wrist and elbow are oriented correctly.")
print("\nSince legs are symmetric but elbows are left/right specific, you have options:")
print("\n1. Use left elbow pattern for all leg joints:")
print(f"   {new_left_leg.tolist()}")
print("\n2. Use right elbow pattern for all leg joints:")
print(f"   {new_right_leg.tolist()}")
print("\n3. Use left pattern for left leg, right pattern for right leg (like arms)")
print(f"   Left leg: {new_left_leg.tolist()}")
print(f"   Right leg: {new_right_leg.tolist()}")
