import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QComboBox,
    QDial,
    QSlider,
    QPushButton,
    QGridLayout,
    QGroupBox,
    QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

channel_names = ["X", "Y", "Z"]

from PyQt6.QtCore import QThread, pyqtSignal
import threading  # On MuJoCo Thread
import json
import os


class MujocoThread(QThread):
    finished = pyqtSignal()  # Signal: MuJoCo complete or stop

    def __init__(self, anim, parser, parent=None):
        super().__init__(parent)
        self.anim = anim
        self.parser = parser
        self.stop_flag = threading.Event()

    def run(self):
        # Running MuJoCo in a thread display
        from mujoco_xsens_bvh_view import mujoco_displayanimanim  # Delayed import to avoid loops

        d = mujoco_displayanimanim(
            self.parser, self.anim
        )  # Assuming bvh_file=None, use pre-computed anim
        
        d._init_xml_data(save_flag=True)  # Using in-memory XML
        try:
            d.animate_bvh()
        except Exception as e:
            print(f"MuJoCo error: {e}")
        finally:
            self.finished.emit()


class OffsetManager:
    """This class is used to read, save, and parse offset data from a JSON file
    Data format ：{ "joint_name": { "X": offset, "Y": offset, "Z": offset }, ... }
    """

    def __init__(self, default_path="offsets.json"):
        self.default_path = default_path
        self.offsets = self.load_offsets()

    def load_offsets(self, path=None):
        """Load the offset from the specified path. If the path does not exist, initialize it to all zeros
        """
        path = path or self.default_path
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading JSON: {e} Initialized to all zeros")
        else:
            print(f"The path {path} does not exist. Initialize it to all zeros")
        return {}  # 返回空字典，后续在窗口中填充全 0

    def save_offsets(self, offsets, path):
        """Save the offset to the specified path"""
        try:
            with open(path, "w") as f:
                json.dump(offsets, f, indent=4)
            print(f"Offset saved to {path}。")
        except IOError as e:
            print(f"Error saving JSON: {e}。")

    def parse_to_window_format(self, joint_names, offsets_dict):
        """Parse JSON data into a window's offsets dictionary format: {(joint_idx, channel_idx): offset}"""
        offsets = {}
        for j, joint in enumerate(joint_names):
            joint_data = offsets_dict.get(joint, {"X": 0.0, "Y": 0.0, "Z": 0.0})
            for c, channel in enumerate(channel_names):
                offsets[(j, c)] = joint_data.get(channel, 0.0)
        return offsets

    def format_for_save(self, offsets, joint_names):
        """Format the window offsets as a JSON file and save it"""
        save_data = {}
        for j, joint in enumerate(joint_names):
            save_data[joint] = {
                channel_names[c]: offsets.get((j, c), 0.0) for c in range(3)
            }
        return save_data


class CurveEditorWindow(QMainWindow):
    def __init__(self, joint_names, data, scale=100.0, parser=None):
        super().__init__()
        self.parser = parser  # Pass in a BVHParser instance
        self.is_frozen = False
        self.mujoco_thread = None
        self.is_mujoco_running = False

        self.setWindowTitle("BVH Curve Editor (Independent Bias per Joint/Channel)")
        self.setGeometry(100, 100, 1200, 800)
        self.joint_num = data.shape[1]
        self.frame_num = data.shape[0]
        self.joint_names = joint_names
        self.data = data
        # Initialize offset dictionary：{ (joint_idx, channel_idx): bias }
        self.offset_manager = OffsetManager(default_path="offsets.json")
        loaded_offsets = self.offset_manager.load_offsets()
        # self.offsets = {(j, c): 0.0 for j in range(self.joint_num) for c in range(3)}
        self.offsets = self.offset_manager.parse_to_window_format(
            joint_names, loaded_offsets
        )

        self.scale = scale

        # Current selection
        self.selected_joint_idx = 0
        self.selected_channel_idx = 0

        # Center widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Control Group
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout(control_group)

        # Joint Selection
        self.joint_combo = QComboBox()
        self.joint_combo.addItems(self.joint_names)
        self.joint_combo.currentIndexChanged.connect(self.on_joint_changed)
        control_layout.addWidget(QLabel("Joint:"), 0, 0)
        control_layout.addWidget(self.joint_combo, 0, 1)

        # Channel Selection
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(channel_names)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        control_layout.addWidget(QLabel("Channel:"), 0, 2)
        control_layout.addWidget(self.channel_combo, 0, 3)

        # offset knob
        self.offset_dial = QDial()
        self.offset_dial.setRange(-1000, 1000)  # 范围-100到100，单位0.1
        self.offset_dial.setNotchesVisible(True)
        self.offset_dial.setWrapping(False)
        self.offset_dial.valueChanged.connect(self.on_offset_changed)
        self.offset_dial.setSingleStep(1)
        control_layout.addWidget(QLabel("Offset Knob:"), 1, 0)
        control_layout.addWidget(self.offset_dial, 1, 1, 1, 2)

        # Offset value display
        self.offset_label = QLabel(f"Offset: {self.offsets[(0, 0)]:.2f}")
        self.offset_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        control_layout.addWidget(self.offset_label, 1, 3)

        # Add path selection UI
        self.path_edit = QLineEdit(self.offset_manager.default_path)
        self.path_edit.setToolTip("编辑或选择 JSON 文件路径")
        control_layout.addWidget(QLabel("JSON Path:"), 2, 0)
        control_layout.addWidget(self.path_edit, 2, 1, 1, 2)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.on_browse_clicked)
        control_layout.addWidget(self.browse_button, 2, 3)

        # Add Button
        self.apply_button = QPushButton("Apply and Preview")
        self.apply_button.clicked.connect(self.on_apply_preview)
        control_layout.addWidget(self.apply_button, 3, 0, 1, 4)

        layout.addWidget(control_group)

        # Matplotlib Figure Canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Initialize axes and plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Rotation Curve")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Rotation Value")
        self.ax.grid(True)
        self.frames = np.arange(self.frame_num)

        # Initial drawings and knob settings
        self.update_plot()
        self.update_dial_from_offset()

        # Add cursor
        self.cursor = Cursor(self.ax, useblit=True, color="red", linewidth=1)

    def freeze_ui(self):
        self.is_frozen = True
        self.joint_combo.setEnabled(False)
        self.channel_combo.setEnabled(False)
        self.offset_dial.setEnabled(False)
        self.apply_button.setEnabled(False)
        print("UI freezed")

    def unfreeze_ui(self):
        self.is_frozen = False
        self.joint_combo.setEnabled(True)
        self.channel_combo.setEnabled(True)
        self.offset_dial.setEnabled(True)
        self.apply_button.setEnabled(True)
        print("UI unfreezed")

    def on_browse_clicked(self):
        """This triggers a file selection dialog box and updates the path text box"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select or create a JSON file",
            self.path_edit.text(),
            "JSON files (*.json);;All files (*)",
        )
        if file_path:
            self.path_edit.setText(file_path)

    def on_apply_preview(self):
        print("button press")
        """Apply and preview: Save the offset to the specified path and exute the preview logic (MuJoCo can be extended later)"""
        save_path = self.path_edit.text()
        if not save_path:
            print("The path is empty, so it cannot be saved")
            return
        save_data = self.offset_manager.format_for_save(self.offsets, self.joint_names)
        self.offset_manager.save_offsets(save_data, save_path)
        print("json has saved: " + save_path)

        if self.is_frozen:
            return
        self.freeze_ui()
        # Complete parsing and start the MuJoCo thread
        rotations = self.get_new_data()  # Get current adjustment data
        print("rotations has add offset")
        positions = np.copy(self.parser.positions)
        _quats, _positions, _offsets, _parents = (
            self.parser._MOTION_data_post_processing(
                rotations, positions, reset_to_zero=True
            )
        )
        print("MOTION_data_post_processing")
        from BVHParser import Anim

        anim = Anim(_quats, _positions, _offsets, _parents, self.joint_names)
        print("anim prepared")
        self.mujoco_thread = MujocoThread(anim, self.parser, self)
        self.mujoco_thread.finished.connect(self.on_mujoco_finished)
        self.mujoco_thread.start()
        self.is_mujoco_running = True

    def get_channel_data(self):
        """Get the data for the currently selected joint and channel, and apply the offset"""
        joint_data = self.data[:, self.selected_joint_idx, self.selected_channel_idx]
        current_offset = self.offsets[
            (self.selected_joint_idx, self.selected_channel_idx)
        ]
        return joint_data + current_offset

    def get_new_data(self):
        new_data = np.zeros_like(self.data)
        joint_offset = np.zeros((self.joint_num, 3))
        for i in range(self.joint_num):
            for j in range(3):
                joint_offset[i, j] = self.offsets[(i, j)]
        new_data = self.data + joint_offset
        return new_data

    def update_plot(self):
        """Update graph"""
        self.ax.clear()
        channel_data = self.get_channel_data()
        current_offset = self.offsets[
            (self.selected_joint_idx, self.selected_channel_idx)
        ]
        self.ax.plot(
            self.frames,
            channel_data,
            "b-",
            linewidth=1,
            label=f"{self.joint_names[self.selected_joint_idx]} {channel_names[self.selected_channel_idx]}",
        )
        self.ax.set_title(
            f"Curve: {self.joint_names[self.selected_joint_idx]} - {channel_names[self.selected_channel_idx]} (Offset: {current_offset:.2f})"
        )
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Rotation Value")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def update_dial_from_offset(self):
        """Update knob position based on current offset"""
        current_offset = self.offsets[
            (self.selected_joint_idx, self.selected_channel_idx)
        ]
        dial_value = int(current_offset * self.scale)  # Convert to knob integer value
        self.offset_dial.blockSignals(True)  # Preventing recursive signals
        self.offset_dial.setValue(dial_value)
        self.offset_dial.blockSignals(False)
        self.offset_label.setText(f"Offset: {current_offset:.2f}")

    def on_joint_changed(self, idx):
        """Joint Selection variation """
        self.selected_joint_idx = idx
        self.update_dial_from_offset()
        self.update_plot()

    def on_channel_changed(self, idx):
        """Channel Selection Changes"""
        self.selected_channel_idx = idx
        self.update_dial_from_offset()
        self.update_plot()

    def on_offset_changed(self, value):
        """Offset knob change: Updates the current joint-channel offset and redraws"""
        # print("Offset knob changes")
        key = (self.selected_joint_idx, self.selected_channel_idx)
        self.offsets[key] = value / self.scale  # Convert to floating point
        # print("Offset knob changes"+f"Offset: {self.offsets[key]:.2f}")
        self.offset_label.setText(f"Offset: {self.offsets[key]:.2f}")
        self.update_plot()
        if self.is_mujoco_running:
            self.stop_mujoco()
            # Recalculate and restart (similar to the on_apply_preview logic)
            self.on_apply_preview()  # Reuse logic, but optimize to avoid recursion

    def stop_mujoco(self):
        if self.mujoco_thread:
            self.mujoco_thread.stop_flag.set()  # Custom stop signal (must be check in the MuJoCo loop)
            self.mujoco_thread.wait()
            self.is_mujoco_running = False

    def on_mujoco_finished(self):
        self.unfreeze_ui()
        self.is_mujoco_running = False

class CurveEditorWindow_02(CurveEditorWindow):
    ...
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CurveEditorWindow()
    window.show()
    sys.exit(app.exec())