from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QSize, QSignalBlocker

from vispy import scene
import numpy as np

import logging

_logger = logging.getLogger(__name__)


class _RoiSelectionTable(QTableWidget):
    def __init__(self, labels, parent=None):
        super().__init__(len(labels), 3)

        self.labels = labels
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setHorizontalHeaderLabels(["Name", "ID", " "])
        self.setSelectionMode(QTableWidget.SingleSelection)
        for i, label in enumerate(self.labels_info):
            checkbox_item = QTableWidgetItem(label["state"])
            label_name_item = QTableWidgetItem(label["name"])
            label_id_item = QTableWidgetItem(str(label["id"]))

            if label["state"]:
                checkbox_item.setCheckState(Qt.Checked)
            else:
                checkbox_item.setCheckState(Qt.Unchecked)

            self.setItem(i, 0, label_name_item)
            self.setItem(i, 1, label_id_item)
            self.setItem(i, 2, checkbox_item)

        self.resizeColumnsToContents()

        self.cellChanged.connect(self._on_checked)

    @property
    def labels_info(self):
        """Extracted vital info for each label"""
        label_colors = np.array([l.color for l in self.labels])

        label_names = np.array([l.name for l in self.labels])

        l_info = []
        for i_label, label_name in enumerate(label_names):
            label_id = i_label if label_names[i_label] != "Unknown" else -1
            label_dict = {
                "name": label_names[i_label],
                "id": label_id,
                "state": self.labels[i_label].is_active,
                "color": label_colors[i_label, :],
                "vertices": self.labels[i_label].vertices,
            }
            l_info.append(label_dict)
        return l_info

    def minimumSizeHint(self):
        w = self.verticalHeader().width() + 22  # +22 is for scrollbar
        for i in range(self.columnCount()):
            w += self.columnWidth(i)
        h = self.horizontalHeader().height()
        for i in range(min(self.rowCount(), 10)):
            h += self.rowHeight(i)
        return QSize(w, h)

    def sizeHint(self):
        return self.minimumSizeHint()

    def _on_checked(self, i):
        _logger.debug("clicked --> %d " % (i + 1))
        self.labels[i].is_active = not self.labels[i].is_active


class RoiSelectionDialog(QDialog):
    def __init__(self, labels, mesh, parent=None):
        super().__init__(parent)
        button_widgets_layout = QVBoxLayout()
        widgets_layout = QHBoxLayout()
        button_widgets_layout.addLayout(widgets_layout)
        self._set_table(labels)
        self.table.cellChanged.connect(self.plot_atlas)
        widgets_layout.addWidget(self.table)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self._on_cancel)
        button_box.button(QDialogButtonBox.Ok).setDefault(True)
        button_widgets_layout.addWidget(button_box)

        self._mesh = mesh
        # self.mesh.color = np.random.rand(len(self.mesh), 1)
        brain_widget = self.create_brain_widget()
        widgets_layout.addWidget(brain_widget)

        self.setLayout(button_widgets_layout)
        self.plot_atlas()

    def _set_table(self, labels):
        self.table = _RoiSelectionTable(labels)

    def create_brain_widget(self):
        canvas = scene.SceneCanvas(keys="interactive", show=True)

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = "turntable"
        view.camera.fov = 50
        view.camera.distance = 400
        # Make light follow the camera
        self._mesh.shared_program.frag["camtf"] = view.camera.transform
        view.add(self._mesh)
        return canvas.native

    def plot_atlas(self):
        # reset colors to white
        self._mesh._alphas[:, :] = 0.0
        self._mesh._alphas_buffer.set_data(self._mesh._alphas)
        labels = self.table.labels
        labels_info = self.table.labels_info
        data_vec = np.zeros((len(self._mesh),), dtype=np.float32)
        mask = np.zeros((len(self._mesh),), dtype=bool)
        # colors = np.ones((len(data_vec), 3))
        # colors = np.ones((1,3))
        colors = np.empty((0, 3))
        for label, label_info in zip(labels, labels_info):
            data_vec[label.vertices] = label_info["id"]
            mask[label.vertices] = True
            colors = np.r_[colors, label_info["color"][:3].reshape([1, 3])]

        kw = {}
        kw["cmap"] = colors
        kw["interpolation"] = "linear"
        kw["vmin"] = 0
        kw["vmax"] = 4
        kw["under"] = [1, 0, 0]
        kw["over"] = [1, 0, 0]
        kw["clim"] = (0, 4)

        if np.any(mask):
            self._mesh.add_overlay(
                data_vec[mask], vertices=np.where(mask)[0], to_overlay=1, **kw
            )
        for label in labels:
            if not label.is_active:
                self._mesh._alphas[label.vertices, 1] = 0.3
        self._mesh._alphas_buffer.set_data(self._mesh._alphas)
        # self._mesh._alphas[]
        self._mesh.update()

        # self._mesh.add_overlay(data_vec[mask], vertices=np.where(mask)[0])

    def _on_ok(self):
        _logger.debug("OK")
        QDialog.accept(self)

    def _on_cancel(self):
        _logger.debug("Cancel")
        QDialog.reject(self)


class _SeedSelectionTable(_RoiSelectionTable):

    def _on_checked(self, i):
        block_signals = QSignalBlocker(self)  # noqa
        _logger.debug("clicked --> %d " % (i + 1))
        for l in self.labels:
            l.is_active = False
        self.labels[i].is_active = not self.labels[i].is_active
        for j in range(self.rowCount()):
            self.item(j, 2).setCheckState(Qt.Unchecked)
        self.item(i, 2).setCheckState(Qt.Checked)


class SeedSelectionDialog(RoiSelectionDialog):

    def _set_table(self, labels):
        self.table = _SeedSelectionTable(labels)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    import os.path as op
    import mne

    stride = 30000

    labels_info = [
        {
            "name": "ROI      #" + str(i),
            "state": False,
            "id": i + 1,
            "color": np.random.rand(1, 3),
            "vertices": np.arange(i * stride, (i + 1) * stride),
        }
        for i in range(4)
    ]

    labels_info[0]["color"] = np.array([[1, 1, 0]])
    labels_info[1]["color"] = np.array([[1, 0, 1]])

    SURF_DIR = op.join(mne.datasets.sample.data_path(), "subjects/sample")

    app = QApplication(sys.argv)
    d = RoiSelectionDialog(labels_info, SURF_DIR)
    d.show()

    sys.exit(app.exec_())
