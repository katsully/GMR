import numpy as np

def quat_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_inverse(q):
    """Inverse of quaternion [w, x, y, z]"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

# Current rotation offsets
old_rotation = np.array([0.70710678, 0.0, 0.0, -0.70710678])  # Legs/hips/torso
left_elbow_rotation = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # Left elbow (target)
right_elbow_rotation = np.array([-0.70710678, 0.70710678, 0.0, 0.0])  # Right elbow

# Calculate the transformation needed
# We want: new_rotation = transformation * old_rotation
# So: transformation = new_rotation * inverse(old_rotation)

# For left side (assuming we want to match left elbow pattern)
transformation_left = quat_multiply(left_elbow_rotation, quat_inverse(old_rotation))
print("Transformation needed (left pattern):")
print(f"  [w, x, y, z] = [{transformation_left[0]:.8f}, {transformation_left[1]:.8f}, {transformation_left[2]:.8f}, {transformation_left[3]:.8f}]")

# Apply this transformation to the old rotation to verify
result_left = quat_multiply(transformation_left, old_rotation)
print(f"\nVerification (should match left elbow): {result_left}")
print(f"Expected: {left_elbow_rotation}")

# For right side
transformation_right = quat_multiply(right_elbow_rotation, quat_inverse(old_rotation))
print("\n\nTransformation needed (right pattern):")
print(f"  [w, x, y, z] = [{transformation_right[0]:.8f}, {transformation_right[1]:.8f}, {transformation_right[2]:.8f}, {transformation_right[3]:.8f}]")

# Apply this transformation to the old rotation to verify
result_right = quat_multiply(transformation_right, old_rotation)
print(f"\nVerification (should match right elbow): {result_right}")
print(f"Expected: {right_elbow_rotation}")

# Now let's think about what rotation to use for center/symmetric joints (pelvis, torso)
# These might need a different pattern - possibly no left/right distinction
print("\n\n=== ANALYSIS ===")
print("Old pattern (legs/hips/torso): 90° rotation around -X axis")
print("Left elbow/wrist pattern: 90° rotation with X+Y components")
print("Right elbow/wrist pattern: Similar but mirrored")
print("\nFor symmetric joints (pelvis, torso), we might want:")
print("  Option 1: Use left pattern (might cause right-bias)")
print("  Option 2: Use right pattern (might cause left-bias)")
print("  Option 3: Use average/intermediate rotation")
print("  Option 4: Keep original rotation")

# Let's also calculate what a "neutral" version would be (average of left and right)
# Average quaternion (simple approach - there are more sophisticated methods)
avg_quat = (left_elbow_rotation + right_elbow_rotation) / 2
avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize
print(f"\nAverage of left and right elbow rotations (normalized):")
print(f"  [w, x, y, z] = [{avg_quat[0]:.8f}, {avg_quat[1]:.8f}, {avg_quat[2]:.8f}, {avg_quat[3]:.8f}]")

# Actually, for center joints, maybe we want something different
# Let's try a 90° rotation around Y axis (another common pattern)
y_rotation = np.array([0.70710678, 0.0, 0.70710678, 0.0])
print(f"\n90° rotation around Y axis:")
print(f"  [w, x, y, z] = [{y_rotation[0]:.8f}, {y_rotation[1]:.8f}, {y_rotation[2]:.8f}, {y_rotation[3]:.8f}]")

# Or 90° rotation around Z axis
z_rotation = np.array([0.70710678, 0.0, 0.0, 0.70710678])
print(f"\n90° rotation around Z axis:")
print(f"  [w, x, y, z] = [{z_rotation[0]:.8f}, {z_rotation[1]:.8f}, {z_rotation[2]:.8f}, {z_rotation[3]:.8f}]")
