"""
Interactive tool for tuning rotation offsets in IK config files.
Displays first frame with human motion arrows and allows real-time rotation adjustment via sliders.
"""

import argparse
import pathlib
import json
import pickle
import numpy as np
import mujoco as mj
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from rich import print
import tkinter as tk
from tkinter import ttk
import threading


def draw_frame(pos, mat, v, size, joint_name=None, orientation_correction=R.from_euler("xyz", [0, 0, 0]), pos_offset=np.array([0, 0, 0])):
    """Draw coordinate frame with orientation correction applied"""
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

    fix = orientation_correction.as_matrix()

    for i in range(3):
        if v.user_scn.ngeom >= v.user_scn.maxgeom:
            print(f"[red]Warning: Max geometry limit reached ({v.user_scn.maxgeom})[/red]")
            return

        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name

        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1


class SliderControlUI:
    def __init__(self, tuner):
        self.tuner = tuner
        print("[cyan]Creating Tkinter window...[/cyan]")
        self.root = tk.Tk()
        self.root.title("IK Offset Tuner - Controls")
        self.root.geometry("400x300")

        # Bring window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        print("[green]Tkinter window created![/green]")

        # Body part selector
        frame_body = ttk.Frame(self.root, padding="10")
        frame_body.pack(fill=tk.X)

        ttk.Label(frame_body, text="Body Part:").pack(side=tk.LEFT)
        self.body_var = tk.StringVar(value=self.tuner.current_body)
        self.body_dropdown = ttk.Combobox(frame_body, textvariable=self.body_var,
                                          values=self.tuner.body_parts, state="readonly", width=30)
        self.body_dropdown.pack(side=tk.LEFT, padx=5)
        self.body_dropdown.bind("<<ComboboxSelected>>", self.on_body_changed)

        # Sliders frame
        frame_sliders = ttk.LabelFrame(self.root, text="Rotation Offsets (degrees)", padding="10")
        frame_sliders.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # X slider
        ttk.Label(frame_sliders, text="X (Roll):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.x_var = tk.DoubleVar(value=self.tuner.euler_offsets[self.tuner.current_body][0])
        self.x_slider = ttk.Scale(frame_sliders, from_=-180, to=180, orient=tk.HORIZONTAL,
                                  variable=self.x_var, command=self.on_slider_change)
        self.x_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.x_label = ttk.Label(frame_sliders, text=f"{self.x_var.get():.1f}°")
        self.x_label.grid(row=0, column=2, padx=5)

        # Y slider
        ttk.Label(frame_sliders, text="Y (Pitch):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.y_var = tk.DoubleVar(value=self.tuner.euler_offsets[self.tuner.current_body][1])
        self.y_slider = ttk.Scale(frame_sliders, from_=-180, to=180, orient=tk.HORIZONTAL,
                                  variable=self.y_var, command=self.on_slider_change)
        self.y_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.y_label = ttk.Label(frame_sliders, text=f"{self.y_var.get():.1f}°")
        self.y_label.grid(row=1, column=2, padx=5)

        # Z slider
        ttk.Label(frame_sliders, text="Z (Yaw):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.z_var = tk.DoubleVar(value=self.tuner.euler_offsets[self.tuner.current_body][2])
        self.z_slider = ttk.Scale(frame_sliders, from_=-180, to=180, orient=tk.HORIZONTAL,
                                  variable=self.z_var, command=self.on_slider_change)
        self.z_slider.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.z_label = ttk.Label(frame_sliders, text=f"{self.z_var.get():.1f}°")
        self.z_label.grid(row=2, column=2, padx=5)

        frame_sliders.columnconfigure(1, weight=1)

        # Buttons frame
        frame_buttons = ttk.Frame(self.root, padding="10")
        frame_buttons.pack(fill=tk.X)

        ttk.Button(frame_buttons, text="Reset", command=self.on_reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_buttons, text="Save Config", command=self.on_save).pack(side=tk.LEFT, padx=5)

    def on_body_changed(self, event=None):
        """Update sliders when body part selection changes"""
        new_body = self.body_var.get()
        self.tuner.current_body = new_body
        self.tuner.current_body_idx = self.tuner.body_parts.index(new_body)

        # Update slider values
        euler = self.tuner.euler_offsets[new_body]
        self.x_var.set(euler[0])
        self.y_var.set(euler[1])
        self.z_var.set(euler[2])

        self.x_label.config(text=f"{euler[0]:.1f}°")
        self.y_label.config(text=f"{euler[1]:.1f}°")
        self.z_label.config(text=f"{euler[2]:.1f}°")

        print(f"[magenta]Switched to: {new_body}[/magenta]")

    def on_slider_change(self, event=None):
        """Update offsets when sliders change"""
        self.tuner.euler_offsets[self.tuner.current_body][0] = self.x_var.get()
        self.tuner.euler_offsets[self.tuner.current_body][1] = self.y_var.get()
        self.tuner.euler_offsets[self.tuner.current_body][2] = self.z_var.get()

        # Update labels
        self.x_label.config(text=f"{self.x_var.get():.1f}°")
        self.y_label.config(text=f"{self.y_var.get():.1f}°")
        self.z_label.config(text=f"{self.z_var.get():.1f}°")

    def on_reset(self):
        """Reset current body to original offsets"""
        self.tuner.reset_current_body()
        euler = self.tuner.euler_offsets[self.tuner.current_body]
        self.x_var.set(euler[0])
        self.y_var.set(euler[1])
        self.z_var.set(euler[2])

    def on_save(self):
        """Save config"""
        self.tuner.save_config()

    def run(self):
        """Run the tkinter main loop"""
        self.root.mainloop()


class InteractiveOffsetTuner:
    def __init__(self, robot_type, config_path, motion_file):
        self.robot_type = robot_type
        self.config_path = pathlib.Path(config_path)
        self.motion_file = pathlib.Path(motion_file)

        # Load config
        with open(self.config_path) as f:
            self.config = json.load(f)

        # Load motion data (should be _inter.pkl file with human motion)
        with open(self.motion_file, "rb") as f:
            motion_data = pickle.load(f)

        self.first_frame = motion_data[0]

        # Convert list format to tuple format: {"body_name": [pos, quat]} -> {"body_name": (pos, quat)}
        self.human_data = {}
        for body_name, data in self.first_frame.items():
            pos = np.array(data[0])
            quat = np.array(data[1])
            self.human_data[body_name] = (pos, quat)

        print(f"[bold cyan]Loaded human motion with {len(self.human_data)} body parts[/bold cyan]")

        # Initialize MuJoCo model and viewer
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]

        # Get list of body parts we can tune
        self.body_parts = list(self.config["ik_match_table1"].keys())
        self.current_body_idx = 0
        self.current_body = self.body_parts[self.current_body_idx]

        # Euler angles for current body (in degrees)
        self.euler_offsets = {}
        for body_name in self.body_parts:
            # Convert existing quaternion offset to euler
            quat_offset = self.config["ik_match_table1"][body_name][4]
            euler = R.from_quat(quat_offset, scalar_first=True).as_euler("xyz", degrees=True)
            self.euler_offsets[body_name] = euler.copy()

        print(f"\n[bold green]Interactive Offset Tuner Started![/bold green]")
        print(f"[yellow]Body parts to tune: {len(self.body_parts)}[/yellow]")

    def reset_current_body(self):
        """Reset current body to original config values"""
        quat_offset = self.config["ik_match_table1"][self.current_body][4]
        euler = R.from_quat(quat_offset, scalar_first=True).as_euler("xyz", degrees=True)
        self.euler_offsets[self.current_body] = euler.copy()
        print(f"[yellow]Reset {self.current_body} to original values[/yellow]")
        print(f"  Euler angles (XYZ): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")

    def save_config(self):
        """Save modified config to new file"""
        # Update config with new quaternions
        for body_name in self.body_parts:
            euler = self.euler_offsets[body_name]
            quat = R.from_euler("xyz", euler, degrees=True).as_quat(scalar_first=True)

            # Update both ik_match_table1 and ik_match_table2
            self.config["ik_match_table1"][body_name][4] = quat.tolist()
            if body_name in self.config.get("ik_match_table2", {}):
                self.config["ik_match_table2"][body_name][4] = quat.tolist()

        # Generate output filename
        output_path = self.config_path.parent / (self.config_path.stem + "_tuned.json")

        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=4)

        print(f"\n[bold green]✓ Config saved to: {output_path}[/bold green]")

    def render_frame(self, viewer, init_camera=False):
        """Render current frame with updated offsets"""
        # Set robot to standing position
        self.data.qpos[:3] = [0, 0, 0.75]
        self.data.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion
        self.data.qpos[7:] = 0
        mj.mj_forward(self.model, self.data)

        # Set camera to look at human data center (only on first frame)
        if init_camera:
            center_body = "Hips" if "Hips" in self.human_data else list(self.human_data.keys())[0]
            center_pos = np.array(self.human_data[center_body][0])
            viewer.cam.lookat[:] = center_pos
            viewer.cam.distance = 2.5  # Closer view for human skeleton
            viewer.cam.azimuth = 90  # Side view
            viewer.cam.elevation = -20  # Look down slightly

        # Clear custom geometry
        viewer.user_scn.ngeom = 0

        # Draw ALL human motion arrows (not just ones in config)
        drawn_count = 0
        for body_name, (pos, quat) in self.human_data.items():
            original_mat = R.from_quat(quat, scalar_first=True).as_matrix()

            # Check if this body has an offset in config
            if body_name in self.euler_offsets:
                euler_offset = self.euler_offsets[body_name]
                orientation_correction = R.from_euler("xyz", euler_offset, degrees=True)

                # Highlight current body being edited
                if body_name == self.current_body:
                    size = 0.25  # Larger arrows for selected body
                    draw_frame(pos, original_mat, viewer, size, joint_name=body_name, orientation_correction=orientation_correction)
                else:
                    size = 0.12
                    draw_frame(pos, original_mat, viewer, size, orientation_correction=orientation_correction)
                drawn_count += 1
            else:
                # Draw bodies not in config with smaller grey arrows
                size = 0.08
                orientation_correction = R.from_euler("xyz", [0, 0, 0], degrees=True)
                draw_frame(pos, original_mat, viewer, size, orientation_correction=orientation_correction)

        # Update the scene
        viewer.sync()

    def run_viewer(self):
        """Run MuJoCo viewer in separate thread"""
        import time
        # Launch MuJoCo viewer
        with mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False
        ) as viewer:
            # Make robot transparent
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1

            print(f"\n[bold cyan]Viewer active! Use sliders to adjust offsets[/bold cyan]")
            print(f"[yellow]Number of body parts with offsets: {len(self.euler_offsets)}[/yellow]")
            print(f"[yellow]Body parts in human_data: {list(self.human_data.keys())[:5]}...[/yellow]")
            print(f"[yellow]Body parts in config: {list(self.euler_offsets.keys())[:5]}...[/yellow]")

            frame_count = 0
            first_frame_debug = True
            while viewer.is_running():
                # Only set camera on first frame
                self.render_frame(viewer, init_camera=(frame_count == 0))

                # Debug output on first frame and then every 100 frames
                if first_frame_debug:
                    print(f"[cyan]First frame, geoms drawn: {viewer.user_scn.ngeom}[/cyan]")
                    first_frame_debug = False
                elif frame_count % 100 == 0:
                    print(f"[dim]Frame {frame_count}, geoms: {viewer.user_scn.ngeom}[/dim]")
                frame_count += 1

                time.sleep(0.016)  # ~60 FPS

        print("[green]Viewer closed[/green]")

    def run(self):
        """Main interaction loop with MuJoCo viewer and slider UI"""
        import time

        print("[cyan]Starting viewer thread...[/cyan]")
        # Launch MuJoCo viewer in separate thread
        viewer_thread = threading.Thread(target=self.run_viewer, daemon=True)
        viewer_thread.start()

        # Small delay to let viewer initialize
        time.sleep(0.5)

        print("[cyan]Starting slider UI on main thread...[/cyan]")
        # Run slider UI on main thread (required for Tkinter on Windows)
        ui = SliderControlUI(self)
        ui.run()  # This blocks until window is closed


def main():
    parser = argparse.ArgumentParser(description="Interactive IK offset tuner")
    parser.add_argument(
        "--motion_file",
        type=str,
        default="data/Z_Solo_Best_robot_inter.pkl",
        help="FBX motion file (pickle) to load first frame from (use _inter.pkl files)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="general_motion_retargeting/ik_configs/xsense_fbx_offline_to_g1.json",
        help="IK config JSON file to tune"
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="unitree_g1",
        help="Robot type"
    )

    args = parser.parse_args()

    tuner = InteractiveOffsetTuner(
        robot_type=args.robot,
        config_path=args.config,
        motion_file=args.motion_file
    )

    tuner.run()


if __name__ == "__main__":
    main()
