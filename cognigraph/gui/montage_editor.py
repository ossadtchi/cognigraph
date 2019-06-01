"""
Montage editor dialog

Exposed classes
---------------
MontageEditor: QDialog

"""
import sys
import os.path as op
from functools import partial
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
from PyQt5 import Qt
from PyQt5.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
    QLabel,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QSplitter,
    QMessageBox,
)
from typing import List
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas

# import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mne
import matplotlib

from cognigraph.utils.channels import save_montage
from cognigraph import MONTAGES_DIR

font = {"weight": "bold", "size": 16}
matplotlib.rc("font", **font)


def _common_suffix(list_of_strings: List[str]):
    reversed_strings = [l[::-1] for l in list_of_strings]
    resversed_suffix = op.commonprefix(reversed_strings)
    return resversed_suffix[::-1]


class _ButtonLineEdit(QWidget):
    def __init__(self, parent=None, init_text=None):
        QWidget.__init__(self, parent)
        self._init_text = init_text
        self._lineedit = QLineEdit(init_text)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._lineedit)
        self.done_button = QPushButton("&Done")
        layout.addWidget(self.done_button)
        self.done_button.clicked.connect(self._on_done)
        self._lineedit.editingFinished.connect(self._on_done)

    def _on_done(self):
        self.hide()
        self._lineedit.setText(self._init_text)

    def show(self):
        QWidget.show(self)
        # self.done_button.setDefault(True)
        # self.done_button.setAutoDefault(True)

    def hide(self):
        QWidget.hide(self)
        # self.done_button.setDefault(False)
        # self.done_button.setAutoDefault(False)


class _DragAndDroppableList(QListWidget):
    item_dropped = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QListWidget.__init__(self, parent)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def sizeHint(self):
        fm = self.fontMetrics()
        # get size of this string to set list height accordingly
        test_string = "EEG001"
        w = fm.boundingRect(test_string).width()
        h = fm.boundingRect(test_string).height()
        tot_h = (self.count() + 1) * h
        return Qt.QSize(w, tot_h)

    def populate_with_montage(self, montage):
        self._clear()
        for i, entry in enumerate(montage.ch_names):
            item = QListWidgetItem(entry)
            item.setData(QtCore.Qt.UserRole, montage.pos[i])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.addItem(item)

    def _clear(self):
        for i in range(self.count()):
            self.takeItem(0)

    def dropEvent(self, *pargs, **kwargs):
        QListWidget.dropEvent(self, *pargs, **kwargs)
        self.item_dropped.emit()


class MontageEditor(QDialog):
    _RED = (255, 39, 0, 200)
    _GREEN = (0, 180, 84, 200)
    _GREY = (240, 240, 240)

    def __init__(self, data_chnames, fwd_montage, parent=None):
        QDialog.__init__(self, parent)
        self.montage_path = None

        self._fwd_montage_orig = fwd_montage
        self._data_chnames = data_chnames

        lists_layout = QHBoxLayout()

        self._data_chnames_list = QListWidget()
        self._data_chnames_list.setSizeAdjustPolicy(1)
        self._data_chnames_list.setSizePolicy(
            Qt.QSizePolicy.Minimum, Qt.QSizePolicy.Expanding
        )
        data_chnames_list_label = QLabel("Data montage")
        data_chnames_list_label.setBuddy(self._data_chnames_list)
        data_chnames_layout = QVBoxLayout()
        data_chnames_layout.addWidget(data_chnames_list_label)
        data_chnames_layout.addWidget(self._data_chnames_list)
        lists_layout.addLayout(data_chnames_layout)

        self._fwd_chnames_list = _DragAndDroppableList()
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        self._fwd_chnames_list.populate_with_montage(fwd_montage)
        self._fwd_chnames_list.itemChanged.connect(self._mark_goods)
        self._fwd_chnames_list.item_dropped.connect(self._mark_goods)
        fwd_chnames_list_label = QLabel("Forward montage")
        fwd_chnames_list_label.setBuddy(self._fwd_chnames_list)
        fwd_chnames_layout = QVBoxLayout()
        fwd_chnames_layout.addWidget(fwd_chnames_list_label)
        fwd_chnames_layout.addWidget(self._fwd_chnames_list)
        lists_layout.addLayout(fwd_chnames_layout)

        align_button = QPushButton("&Align")
        align_button.clicked.connect(self._align_lists)
        reset_button = QPushButton("&Reset")
        reset_button.clicked.connect(self._reset_fwd_chnames)
        sort_button = QPushButton("&Sort")
        sort_button.clicked.connect(self._sort_fwd_chnames)
        add_prefix_button = QPushButton("&Add prefix")
        add_prefix_button.clicked.connect(
            partial(self._bulk_edit, mode="prepend")
        )
        add_suffix_button = QPushButton("&Add suffix")
        add_suffix_button.clicked.connect(
            partial(self._bulk_edit, mode="append")
        )
        del_prefix_button = QPushButton("&Del prefix")
        del_prefix_button.clicked.connect(self._del_prefix)
        del_suffix_button = QPushButton("&Del suffix")
        del_suffix_button.clicked.connect(self._del_suffix)

        adopt_data_chnames_button = QPushButton(">>")
        adopt_data_chnames_button.clicked.connect(self._on_adopt_data_chnames)

        buttons_layout = QVBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(align_button)
        buttons_layout.addWidget(reset_button)
        buttons_layout.addWidget(sort_button)
        buttons_layout.addWidget(add_prefix_button)
        buttons_layout.addWidget(del_prefix_button)
        buttons_layout.addWidget(add_suffix_button)
        buttons_layout.addWidget(del_suffix_button)
        buttons_layout.addWidget(adopt_data_chnames_button)
        buttons_layout.addStretch()
        lists_layout.addLayout(buttons_layout)

        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout()
        canvas_widget.setLayout(canvas_layout)
        self._kind_lineedit = QLineEdit(
            "custom_" + self._fwd_montage_orig.kind
        )
        self._kind_lineedit.setAlignment(QtCore.Qt.AlignHCenter)
        self._kind_lineedit.setFrame(False)
        canvas_layout.addWidget(self._kind_lineedit)

        self._canvas = FigureCanvas(Figure(figsize=(16, 16)))
        self._axes = self._canvas.figure.subplots()
        self._plot_montage(self._fwd_montage_all)
        self._canvas.setSizePolicy(
            Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Expanding
        )
        canvas_layout.addWidget(self._canvas)
        lists_widget = QWidget()
        lists_widget.setLayout(lists_layout)
        splitter = QSplitter()
        splitter.addWidget(lists_widget)
        splitter.addWidget(canvas_widget)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(splitter)

        self._ledit_widget = _ButtonLineEdit(init_text="<suffix>", parent=self)
        layout.addWidget(self._ledit_widget)
        self._ledit_widget.hide()

        self._dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._dialog_buttons.rejected.connect(self.reject)
        self._dialog_buttons.accepted.connect(self._on_ok)
        self._dialog_buttons.button(QDialogButtonBox.Cancel).setAutoDefault(
            False
        )
        layout.addWidget(self._dialog_buttons)

        for data_chname in data_chnames:
            data_item = QListWidgetItem(data_chname)
            self._data_chnames_list.addItem(data_item)

        self.show()
        self._mark_goods()

    def _plot_montage(self, montage):
        colors = np.array(
            [[c for c in self._GREY]] * len(self._fwd_chnames_all), dtype=float
        )
        for i, data_chname in enumerate(self._data_chnames):
            if data_chname == self._fwd_chnames_selected[i]:
                colors[i] = np.array(self._GREEN)[:3]
            else:
                colors[i] = np.array(self._RED)[:3]
        colors /= 255
        # print(colors.shape)

        fig = mne.viz.utils._plot_sensors(
            self._fwd_montage_all.get_pos2d(),
            colors=colors,
            bads=[],
            ch_names=self._fwd_chnames_all,
            title="",
            show_names=True,
            ax=self._axes,
            show=False,
            select=False,
            block=False,
            to_sphere=False,
        )
        fig.axes[0].collections[0].set_sizes([200])

    @property
    def _fwd_chnames_all(self):
        return [
            self._fwd_chnames_list.item(i).text()
            for i in range(self._fwd_chnames_list.count())
        ]

    @property
    def _fwd_montage_all(self):
        """Original montage with renamed channels"""
        ch_names = []
        pos = []
        for i in range(self._fwd_chnames_list.count()):
            ch_names.append(self._fwd_chnames_list.item(i).text())
            pos.append(self._fwd_chnames_list.item(i).data(QtCore.Qt.UserRole))
        pos = np.array(pos)
        lpa = self._fwd_montage_orig.lpa
        rpa = self._fwd_montage_orig.rpa
        nasion = self._fwd_montage_orig.nasion
        kind = "custom_" + self._fwd_montage_orig.kind
        return mne.channels.montage.Montage(
            pos,
            ch_names,
            nasion=nasion,
            lpa=lpa,
            rpa=rpa,
            kind=kind,
            selection=np.arange(len(self._fwd_chnames_all)),
        )

    @property
    def _fwd_chnames_selected(self):
        ch_names = []
        for i, data_chname in enumerate(self._data_chnames):
            ch_names.append(self._fwd_chnames_list.item(i).text())
        return ch_names

    @property
    def _fwd_pos_selected(self):
        pos = []
        for i, data_chname in enumerate(self._data_chnames):
            pos.append(self._fwd_chnames_list.item(i).data(QtCore.Qt.UserRole))
        nasion = self._fwd_montage_orig.nasion
        lpa = self._fwd_montage_orig.lpa
        rpa = self._fwd_montage_orig.rpa
        pos.append(nasion)
        pos.append(lpa)
        pos.append(rpa)
        return np.array(pos)

    @property
    def fwd_montage_selected(self):
        ch_names = self._fwd_chnames_selected
        pos = self._fwd_pos_selected

        nasion = self._fwd_montage_orig.nasion
        if type(nasion) is np.ndarray:
            ch_names.append("Nz")
        rpa = self._fwd_montage_orig.rpa
        if type(rpa) is np.ndarray:
            ch_names.append("RPA")
        lpa = self._fwd_montage_orig.lpa
        if type(lpa) is np.ndarray:
            ch_names.append("LPA")
        kind = self._kind_lineedit.text()
        return mne.channels.montage.Montage(
            pos,
            ch_names,
            nasion=nasion,
            lpa=lpa,
            rpa=rpa,
            kind=kind,
            selection=np.arange(len(self._data_chnames)),
        )

    def _align_lists(self):
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        for i, data_chname in enumerate(self._data_chnames):
            if (
                data_chname in self._fwd_chnames_all
                and data_chname != self._fwd_chnames_all[i]
            ):
                ind_fwd = self._fwd_chnames_all.index(data_chname)
                ind_data = self._data_chnames.index(data_chname)
                fwd_item = self._fwd_chnames_list.takeItem(ind_fwd)
                self._fwd_chnames_list.insertItem(ind_data, fwd_item)
        self._mark_goods()

    def _mark_goods(self):
        self._ok_to_close = True
        self._dialog_buttons.button(QDialogButtonBox.Ok).setDisabled(False)
        for i in range(self._fwd_chnames_list.count()):
            if i < len(self._data_chnames):
                data_chname = self._data_chnames[i]
            else:
                data_chname = None
            if data_chname:
                if self._fwd_chnames_list.item(i).text() == data_chname:
                    self._fwd_chnames_list.item(i).setBackground(
                        QColor(*self._GREEN)
                    )
                else:
                    self._fwd_chnames_list.item(i).setBackground(
                        QColor(*self._RED)
                    )
                    self._ok_to_close = False
                    self._dialog_buttons.button(
                        QDialogButtonBox.Ok
                    ).setDisabled(True)
            else:
                self._fwd_chnames_list.item(i).setBackground(QtCore.Qt.white)

        self._axes.clear()
        self._plot_montage(self._fwd_montage_all)
        self._axes.figure.canvas.draw()

    def _sort_fwd_chnames(self):
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        self._fwd_chnames_list.sortItems()
        self._mark_goods()

    def _bulk_edit(self, mode):
        self._ledit_widget._lineedit.textEdited.connect(
            partial(self._modify_selection, mode=mode)
        )
        self._ledit_widget.show()
        self._ledit_widget._lineedit.setFocus()
        self._ledit_widget._lineedit.selectAll()
        self._fwd_chnames_backup = self._fwd_chnames_all.copy()
        self._modify_selection(self._ledit_widget._lineedit.text(), mode=mode)

    def _modify_selection(self, new_text, mode):
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        if self._fwd_chnames_list.selectedIndexes():
            sel_inds = [
                i.row() for i in self._fwd_chnames_list.selectedIndexes()
            ]
        else:
            sel_inds = range(self._fwd_chnames_list.count())
        for i in sel_inds:
            cur_text = self._fwd_chnames_backup[i]
            if mode == "append":
                text_upd = cur_text + new_text
            elif mode == "prepend":
                text_upd = new_text + cur_text
            if len(text_upd) < 15:  # mne-python restriction
                self._fwd_chnames_list.item(i).setText(text_upd)
        self._mark_goods()

    def _reset_fwd_chnames(self):
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        self._fwd_chnames_list.populate_with_montage(self._fwd_montage_orig)
        self._mark_goods()

    def _del_prefix(self):
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        if self._fwd_chnames_list.selectedIndexes():
            sel_inds = [
                i.row() for i in self._fwd_chnames_list.selectedIndexes()
            ]
        else:
            sel_inds = range(self._fwd_chnames_list.count())
        ch_names = [self._fwd_chnames_list.item(i).text() for i in sel_inds]
        if len(ch_names) == 1:
            return
        commonprefix = op.commonprefix(ch_names)
        for ind, name in zip(sel_inds, ch_names):
            self._fwd_chnames_list.item(ind).setText(name[len(commonprefix) :])
        self._mark_goods()

    def _del_suffix(self):
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        if self._fwd_chnames_list.selectedIndexes():
            sel_inds = [
                i.row() for i in self._fwd_chnames_list.selectedIndexes()
            ]
        else:
            sel_inds = range(self._fwd_chnames_list.count())
        ch_names = [self._fwd_chnames_list.item(i).text() for i in sel_inds]
        if len(ch_names) == 1:
            return
        commonsuffix = _common_suffix(ch_names)
        for ind, name in zip(sel_inds, ch_names):
            self._fwd_chnames_list.item(ind).setText(
                name[: len(name) - len(commonsuffix)]
            )
        self._mark_goods()

    def _on_adopt_data_chnames(self):
        block_signals = QtCore.QSignalBlocker(self._fwd_chnames_list)  # noqa
        for i, data_chname in enumerate(self._data_chnames):
            self._fwd_chnames_list.item(i).setText(data_chname)
        self._mark_goods()

    def _on_ok(self):
        try:
            self.montage_path = save_montage(
                self.fwd_montage_selected, MONTAGES_DIR
            )
            self.accept()
        except FileExistsError as e:
            ans = QMessageBox.question(
                self,
                "Destination file exists",
                str(e) + " Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans == QMessageBox.Yes:
                self.montage_path = save_montage(
                    self.fwd_montage_selected, MONTAGES_DIR, overwrite=True
                )
                self.accept()
            else:
                self.reject()


if __name__ == "__main__":
    data_chnames = ["EEG" + str(i).zfill(3) for i in range(1, 32)]

    m = mne.channels.read_montage("mgh60")
    fwd_chnames = m.ch_names
    app = QApplication(sys.argv)
    window = MontageEditor(data_chnames, m)

    sys.exit(app.exec_())
