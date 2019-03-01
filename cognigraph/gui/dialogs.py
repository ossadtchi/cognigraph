"""Dialogs for cognigraph parameters setup
author: dmalt
date: 2019-02-19

"""
import os
import os.path as op
import inspect
import json
from shutil import copyfile
import numpy as np


import mne
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QComboBox,
                             QDialogButtonBox, QPushButton, QFileDialog,
                             QRadioButton, QGroupBox, QVBoxLayout, QLineEdit,
                             QWidget, QDoubleSpinBox, QProgressDialog,
                             QMessageBox)

from PyQt5.Qt import QSizePolicy, QTimer
from PyQt5.QtCore import Qt, pyqtSignal, QSignalBlocker, QThread, QEventLoop
from cognigraph import COGNIGRAPH_ROOT

import logging

logger = logging.getLogger(name=__file__)


class ResettableComboBox(QComboBox):
    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)

    def setItems(self, items):
        block_signals = QSignalBlocker(self)  # noqa
        for i in range(self.count()):
            self.removeItem(0)
        self.addItems(items)

    def getItems(self):
        return [self.itemText(i) for i in range(self.count())]


class PathSelectorWidget(QWidget):
    """QLineEdit + QPushButton connected to QFileDialog to set path"""
    def __init__(self, dialog_caption, path='', parent=None):
        super().__init__(parent)
        self.caption = dialog_caption

        layout = QHBoxLayout()
        self.path_ledit = QLineEdit(path, readOnly=True)
        # Setup minimum lineedit_width so the path fits in
        fm = self.path_ledit.fontMetrics()
        min_lineedit_width = fm.boundingRect(COGNIGRAPH_ROOT).width()
        self.path_ledit.setMinimumWidth(min_lineedit_width)
        self.browse_button = QPushButton('Browse')
        self.browse_button.setDefault(False)
        self.browse_button.setAutoDefault(False)
        self.browse_button.clicked.connect(self._on_browse_clicked)
        layout.addWidget(self.path_ledit)
        layout.addWidget(self.browse_button)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.file_dialog = QFileDialog(caption=dialog_caption,
                                       directory=self.path_ledit.text())

    def _on_browse_clicked(self):
        if self.path:
            self.path_ledit.setText(self.path)


class FolderSelectorWidget(PathSelectorWidget):
    """
    QLineEdit + QPushButton connected to QFileDialog to set path to folder

    """

    def _on_browse_clicked(self):
        self.path = self.file_dialog.getExistingDirectory()
        super()._on_browse_clicked()


class FileSelectorWidget(PathSelectorWidget):
    """
    QLineEdit + QPushButton connected to QFileDialog to set path to file

    """
    def __init__(self, dialog_caption, file_filter, path='', parent=None):
        super().__init__(dialog_caption, path, parent)
        self.filter = file_filter

    def _on_browse_clicked(self):
        self.path = self.file_dialog.getOpenFileName(filter=self.filter)[0]
        super()._on_browse_clicked()


class StateAwareGroupbox(QGroupBox):
    """QgroupBox tracking valid-invalid state"""
    state_changed = pyqtSignal()

    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self._is_valid = True
        self._is_active = True

    @property
    def is_valid(self):
        """Represents validity of all input parameters"""
        return self._is_valid

    @is_valid.setter
    def is_valid(self, value):
        before = self.is_good
        self._is_valid = value
        after = self.is_good
        if before != after:
            self.state_changed.emit()

    @property
    def is_active(self):
        """Represents disabled-enabled state of the groupbox"""
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        before = self.is_good
        if value:
            self.show()
        else:
            self.hide()
        self._is_active = value
        after = self.is_good
        if before != after:
            self.state_changed.emit()

    @property
    def is_good(self):
        if self.is_active:
            return self.is_valid
        else:
            return True


class ForwardOptionsRadioButtons(QGroupBox):
    """Groupbox with three radio buttons"""
    def __init__(self, title='Forward model', parent=None):
        super().__init__(title=title, parent=parent)

        self.use_default_fwd_radio = QRadioButton('&Use available')
        self.use_default_fwd_radio.setChecked(True)

        self.compute_fwd_radio = QRadioButton('&Compute forward')

        self.import_fwd_radio = QRadioButton('&Import forward')

        forward_radio_layout = QVBoxLayout()
        forward_radio_layout.addWidget(self.use_default_fwd_radio)
        forward_radio_layout.addWidget(self.compute_fwd_radio)
        forward_radio_layout.addWidget(self.import_fwd_radio)

        self.setLayout(forward_radio_layout)


class AnatomyGroupbox(StateAwareGroupbox):
    """Groupbox to setup subjects_dir and subject"""
    def __init__(self, title='Anatomy',
                 default_subj_dir=op.join(COGNIGRAPH_ROOT, 'data/anatomy'),
                 default_subj='fsaverage', parent=None):
        super().__init__(parent, title=title)
        self.default_subj_dir = default_subj_dir
        self.default_subj = default_subj
        self.subjects_dir = self.default_subj_dir
        self.subject = self.default_subj
        self.subjects = self._get_fsf_subjects(self.subjects_dir)
        # ------------------ anatomy gpbox ------------------ #
        self.use_anatomy_default_radio = QRadioButton('&Use default')
        self.use_anatomy_default_radio.setChecked(True)
        self.use_anatomy_default_radio.toggled.connect(self._on_toggled)
        self.use_anatomy_custom_radio = QRadioButton('&Use custom')
        self.use_anatomy_custom_radio.setChecked(False)

        # self.subject_combobox = ResettableComboBox(label_text='subject:',
        #                                            layout_type='horizontal')
        self.subject_combobox = ResettableComboBox()
        self.subject_combobox.addItems(self.subjects)
        self.subject_combobox.setCurrentText(self.subject)
        subject_label = QLabel('Subject:')
        subject_layout = QHBoxLayout()
        subject_layout.addWidget(subject_label)
        subject_layout.addWidget(self.subject_combobox)

        self.subjects_dir_widget = FolderSelectorWidget(
            'Select subjects directory', path=self.subjects_dir)

        subjects_dir_subject_layout = QVBoxLayout()
        subjects_dir_subject_layout.addWidget(self.subjects_dir_widget)
        subjects_dir_subject_layout.addLayout(subject_layout)

        self.subjects_dir_widget.path_ledit.textChanged.connect(
            self._on_anatomy_path_changed)

        self.anat_path_widget = QWidget()
        self.anat_path_widget.setLayout(subjects_dir_subject_layout)
        self.anat_path_widget.setSizePolicy(QSizePolicy.Minimum,
                                            QSizePolicy.Fixed)
        self.anat_path_widget.setDisabled(True)

        anatomy_gbox_layout = QVBoxLayout()
        anatomy_gbox_layout.addWidget(self.use_anatomy_default_radio)
        anatomy_gbox_layout.addWidget(self.use_anatomy_custom_radio)
        anatomy_gbox_layout.addWidget(self.anat_path_widget)

        self.setLayout(anatomy_gbox_layout)
        # ------------------------------------------------------ #

    def _on_toggled(self):
        if self.use_anatomy_default_radio.isChecked():
            self.anat_path_widget.setDisabled(True)
            self.subjects_dir = self.default_subj_dir
            self.subject = self.default_subj
            self.is_valid = True
        else:
            if self.subjects_dir and self.subject:
                self.is_valid = True
            else:
                self.is_valid = False
            self.anat_path_widget.setDisabled(False)

    def _get_fsf_subjects(self, path):
        files = os.listdir(path)
        return sorted([f for f in files if op.isdir(op.join(path, f))
                       and 'surf' in os.listdir(op.join(path, f))])

    def _on_anatomy_path_changed(self):
        new_path = self.subjects_dir_widget.path_ledit.text()
        if new_path != self.subjects_dir:
            self.subjects = self._get_fsf_subjects(new_path)
            self.subject_combobox.setItems(sorted(self.subjects))
        self.subjects_dir = new_path
        # If valid freesurfer subjects were found:
        if self.subject_combobox.currentText():
            self.subjects_dir = self.subjects_dir
            self.subject = self.subject_combobox
            self.is_valid = True
        else:
            self.is_valid = False


class FwdGeomGroupbox(StateAwareGroupbox):
    """Groupbox to setup spacing and montage"""
    default_montage = 'standard_1005'
    default_spacing = 'oct6'

    def __init__(self, default_subj,
                 title='Select montage and spacing', parent=None):
        super().__init__(title=title, parent=parent)

        # -------- setup paths and fetch montages -------- #
        self._get_builtin_montages()
        self.default_forwards_path = op.join(
            COGNIGRAPH_ROOT, 'data/forwards', default_subj)
        self._get_available_forwards()
        self._is_montages_combo_connected = False  # to spacings combo
        # ------------------------------------------------ #

        # -------- setup comboboxes -------- #
        self.montages_combo = ResettableComboBox()
        montages_combo_label = QLabel('Select montage:')
        montages_combo_label.setBuddy(self.montages_combo)
        montages_combo_layout = QHBoxLayout()
        montages_combo_layout.addWidget(montages_combo_label)
        montages_combo_layout.addWidget(self.montages_combo)
        self.montages_combo_widget = QWidget()
        self.montages_combo_widget.setLayout(montages_combo_layout)

        self.spacings_combo = ResettableComboBox()
        spacings_label = QLabel('Select spacing:')
        spacings_label.setBuddy(self.spacings_combo)
        spacings_combo_layout = QHBoxLayout()
        spacings_combo_layout.addWidget(spacings_label)
        spacings_combo_layout.addWidget(self.spacings_combo)
        self.spacings_combo_widget = QWidget()
        self.spacings_combo_widget.setLayout(spacings_combo_layout)

        self.forwards_combo = ResettableComboBox()
        forwards_label = QLabel('Select forward operator:')
        forwards_label.setBuddy(self.spacings_combo)
        forwards_combo_layout = QVBoxLayout()
        forwards_combo_layout.addWidget(forwards_label)
        forwards_combo_layout.addWidget(self.forwards_combo)
        self.forwards_combo_widget = QWidget()
        self.forwards_combo_widget.setLayout(forwards_combo_layout)
        # ---------------------------------- #

        self.use_default_montage_radio = QRadioButton('&Use Default')
        self.import_montage_radio = QRadioButton('&Import montage')
        self.use_default_montage_radio.setChecked(True)
        self.use_default_montage_radio.toggled.connect(
            self._on_montage_radio_toggled)
        self.import_montage_radio.toggled.connect(
            self._on_montage_radio_toggled)

        self.select_montage_dialog = FileSelectorWidget(
            dialog_caption='Select montage model', path='',
            file_filter='*.txt *.elc *.csd *.elp *.htps'
                        ' *.sfp *.loc *.locs *.eloc *.bvef')
        self.select_montage_dialog.path_ledit.textChanged.connect(
            self._on_montage_path_changed)
        self.select_montage_dialog.setDisabled(True)

        # initialize combos with available forwards
        self.is_use_available = True

        # -------- setup layout -------- #
        default_fwd_layout = QVBoxLayout()
        default_fwd_layout.addWidget(self.use_default_montage_radio)
        default_fwd_layout.addWidget(self.montages_combo_widget)
        default_fwd_layout.addWidget(self.import_montage_radio)
        default_fwd_layout.addWidget(self.select_montage_dialog)
        default_fwd_layout.addWidget(self.spacings_combo_widget)
        default_fwd_layout.addWidget(self.forwards_combo_widget)

        self.default_fwd_widget = QWidget()
        self.default_fwd_widget.setLayout(default_fwd_layout)
        self.default_fwd_widget.setSizePolicy(QSizePolicy.Minimum,
                                              QSizePolicy.Fixed)

        default_fwd_setup_layout = QVBoxLayout()
        default_fwd_setup_layout.addWidget(self.default_fwd_widget)

        self.setLayout(default_fwd_setup_layout)
        # ------------------------------ #

    def _get_builtin_montages(self):
        montages_desc_path = op.join(COGNIGRAPH_ROOT,
                                     'data/montages_desc.json')
        with open(montages_desc_path, 'r') as f:
            description = json.load(f)
            self.montages_desc = description['montages']
            self.spacings_desc = description['spacings']
        self.builtin_montage_names = sorted(
            mne.channels.get_builtin_montages())

    def _get_available_forwards(self):
        p = self.default_forwards_path
        self.available_montages = sorted(
            [i for i in os.listdir(p) if op.isdir(op.join(p, i))])
        self.available_forwards = {}
        for a in self.available_montages:
            spacings = os.listdir(op.join(p, a))
            self.available_forwards[a] = {}
            for s in spacings:
                forwards = [f for f in os.listdir(op.join(p, a, s))
                            if f.endswith('fwd.fif')]
                self.available_forwards[a][s] = forwards

    def _set_hints(self):
        """Set popup description messages to comboboxes"""
        block_signals = QSignalBlocker(self.montages_combo)  # noqa

        for i, key in enumerate(self.montages_combo.getItems()):
            try:
                self.montages_combo.setItemData(
                    i, self.montages_desc[key], Qt.ToolTipRole)
            except KeyError:
                pass

        for i, key in enumerate(self.spacings_combo.getItems()):
            try:
                self.spacings_combo.setItemData(
                    i, self.spacings_desc[key], Qt.ToolTipRole)
            except KeyError:
                pass

    def _set_combos_to_builtin(self):
        """Used when we switch to compute forward mode"""
        # disconnect _on_montage_changed if it was connected
        self.is_montages_combo_connected = False
        self.montages_combo.setItems(self.builtin_montage_names)
        self.montages_combo.setCurrentText(self.default_montage)
        self.spacings_combo.setItems(
            [k for k in self.spacings_desc if k != 'imported'])
        self.spacings_combo.setCurrentText(self.default_spacing)
        self.forwards_combo_widget.hide()
        self._set_hints()

        self.use_default_montage_radio.show()
        self.import_montage_radio.show()
        self.select_montage_dialog.show()

    def _set_combos_to_available(self):
        """Default option: load forward from cognigraph folders structure"""
        block_signals = QSignalBlocker(self.montages_combo)  # noqa

        self.montages_combo.setItems(self.available_montages)
        self.montages_combo.setCurrentText(self.default_montage)

        cur_montage = self.montages_combo.currentText()
        self.spacings_combo.setItems(
            self.available_forwards[cur_montage].keys())
        self.spacings_combo.setCurrentText(self.default_spacing)
        cur_spacing = self.spacings_combo.currentText()

        self.forwards_combo_widget.show()
        self.forwards_combo.setItems(
            self.available_forwards[cur_montage][cur_spacing])

        self.is_montages_combo_connected = True
        self._set_hints()
        self.use_default_montage_radio.hide()
        self.import_montage_radio.hide()
        self.select_montage_dialog.hide()
        self.montage = self.montages_combo.currentText()

    def _on_montage_changed(self):
        if self.is_use_available:
            cur_montage = self.montages_combo.currentText()
            self.spacings_combo.setItems(
                self.available_forwards[cur_montage].keys())
            self.spacings_combo.setCurrentText(self.default_spacing)
            self.forwards_combo.setItems(
                self.available_forwards[cur_montage][
                    self.spacings_combo.currentText()])

            if self.forwards_combo.currentText():
                self.is_valid = True
            else:
                self.is_valid = False

    def _on_montage_radio_toggled(self):
        if self.use_default_montage_radio.isChecked():
            self.select_montage_dialog.setDisabled(True)
            self.montages_combo_widget.setDisabled(False)
            self.montage = self.montages_combo.currentText()
            if self.forwards_combo.currentText():
                self.is_valid = True
            else:
                self.is_valid = False
        else:
            self.select_montage_dialog.setDisabled(False)
            self.montages_combo_widget.setDisabled(True)
            if self.select_montage_dialog.path_ledit.text():
                self.is_valid = True
                self.montage = self.select_montage_dialog.path_ledit.text()
            else:
                self.is_valid = False

    def _on_montage_path_changed(self):
        if self.select_montage_dialog.path_ledit.text():
            self.is_valid = True
            self.montage = self.select_montage_dialog.path_ledit.text()

    @property
    def is_use_available(self):
        return self._is_use_available

    @is_use_available.setter
    def is_use_available(self, value):
        self._is_use_available = value
        if value:
            self._set_combos_to_available()
        else:
            self._set_combos_to_builtin()

    @property
    def is_montages_combo_connected(self):
        return self._is_montages_combo_connected

    @is_montages_combo_connected.setter
    def is_montages_combo_connected(self, value):
        if not self._is_montages_combo_connected and value:
            self.montages_combo.currentTextChanged.connect(
                self._on_montage_changed)
            self._is_montages_combo_connected = value
        elif self._is_montages_combo_connected and not value:
            self.montages_combo.currentTextChanged.disconnect(
                self._on_montage_changed)
            self._is_montages_combo_connected = value


class ImportForwardGroupbox(StateAwareGroupbox):
    """Groupbox for loading forward model from file"""
    def __init__(self, title='Import forward model', parent=None):
        super().__init__(title=title, parent=parent)
        self.is_valid = False
        self.is_active = False
        self.select_fwd_dialog = FileSelectorWidget(
            dialog_caption='Select forward model', path='',
            file_filter='*fwd.fif')

        self.select_fwd_dialog.path_ledit.textChanged.connect(
            self._on_path_changed)

        import_fwd_layout = QVBoxLayout()
        import_fwd_layout.addWidget(self.select_fwd_dialog)

        self.setLayout(import_fwd_layout)
        self.setVisible(True)

    def _on_path_changed(self):
        self.is_valid = True


class ComputeForwardGroupbox(StateAwareGroupbox):
    """Groupbox with parameters for forward computation"""

    def __init__(self, title='Compute forward', parent=None):
        super().__init__(title=title, parent=parent)

        self.default_coreg_file = None
        self._is_valid = True
        self.is_active = False
        # -------- create widgets -------- #
        self.no_coreg_radio = QRadioButton('None')
        self.select_coreg_radio = QRadioButton('Select')
        self.select_coreg_widget = FileSelectorWidget(
            dialog_caption='Select coregistration file',
            file_filter='*trans.fif')

        cond_defaults = self._get_default_mne_conductivities()
        self.brain_conductivity_spinbox = QDoubleSpinBox(
            singleStep=0.001, decimals=3, value=cond_defaults[0])

        self.skull_conductivity_spinbox = QDoubleSpinBox(
            singleStep=0.001, decimals=3, value=cond_defaults[1])

        self.scalp_conductivity_spinbox = QDoubleSpinBox(
            singleStep=0.001, decimals=3, value=cond_defaults[2])

        conductivity_label = QLabel('Conductivities:')
        conductivity_label.setBuddy(self.brain_conductivity_spinbox)
        # -------------------------------- #

        # -------- connect and setup widgets -------- #
        self.no_coreg_radio.setChecked(True)
        self.no_coreg_radio.toggled.connect(self._on_toggled)
        self.select_coreg_radio.toggled.connect(self._on_toggled)
        self.select_coreg_widget.setDisabled(True)

        self.select_coreg_widget.path_ledit.textChanged.connect(
            self._on_path_changed)
        # ------------------------------------------------- #

        # Coregistration subgpbox
        coreg_layout = QVBoxLayout()
        coreg_layout.addWidget(self.no_coreg_radio)
        coreg_layout.addWidget(self.select_coreg_radio)
        coreg_layout.addWidget(self.select_coreg_widget)

        coreg_gpbox = QGroupBox('Coregistration file')
        coreg_gpbox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        coreg_gpbox.setLayout(coreg_layout)
        ##

        conductivity_layout = QHBoxLayout()
        conductivity_layout.addWidget(conductivity_label)
        conductivity_layout.addWidget(self.brain_conductivity_spinbox)
        conductivity_layout.addWidget(self.skull_conductivity_spinbox)
        conductivity_layout.addWidget(self.scalp_conductivity_spinbox)

        compute_fwd_layout = QVBoxLayout()
        compute_fwd_layout.addWidget(coreg_gpbox)
        compute_fwd_layout.addLayout(conductivity_layout)
        self.setLayout(compute_fwd_layout)

    def _get_default_mne_conductivities(self):
        sign = inspect.signature(mne.make_bem_model)
        return sign.parameters['conductivity'].default

    def _on_toggled(self):
        if self.no_coreg_radio.isChecked():
            self.select_coreg_widget.setDisabled(True)
            self.is_valid = True
        else:
            self.select_coreg_widget.setDisabled(False)
            if self.select_coreg_widget.path_ledit.text():
                self.is_valid = True
            else:
                self.is_valid = False

    def _on_path_changed(self):
        if self.select_coreg_widget.path_ledit.text():
            self.is_valid = True


class BadInputFile(Exception):
    pass


class ComputeForwardInThread(QThread):
    """Compute forward model in parallel thread"""
    progress = pyqtSignal(int)
    exception_ocurred = pyqtSignal(Exception)

    def __init__(self, montage, subjects_dir, subject,
                 spacing, conductivity, trans_file, dest_dir, n_jobs=8,
                 verbose=True, parent=None):
        super().__init__(parent)
        self.montage = montage
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.spacing = spacing
        self.conductivity = conductivity
        self.trans_file = trans_file
        self.dest_dir = dest_dir
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.is_successful = True
        self.fwd_savename = None

    def no_blocking_execution(self):
        """
        Execution of parallel thread with main thread waiting for it to finish
        by locking its interface for input but still keeping it alive

        """
        q = QEventLoop()
        # -------- setup progress dialog -------- #
        progress_dialog = QProgressDialog()
        progress_dialog.setLabelText(
            'Computing forward model... Be patient.')
        progress_dialog.setCancelButtonText(None)

        progress_dialog.setRange(0, 100)
        progress_dialog.show()
        self.parent().setDisabled(True)
        # --------------------------------------- #
        self.progress.connect(progress_dialog.setValue)
        self.exception_ocurred.connect(self._on_fwd_comp_exception)
        self.finished.connect(q.quit)
        self.start()
        q.exec(QEventLoop.ExcludeUserInputEvents)
        progress_dialog.hide()
        self.parent().setDisabled(False)
        return self.is_successful

    def _on_fwd_comp_exception(self, exception):
        msg = QMessageBox(self.parent())
        msg.setText('Forward computation failed')
        msg.setDetailedText(str(exception))
        msg.setIcon(QMessageBox.Warning)
        msg.show()
        self.is_successful = False

    def run(self):
        """Compute 3-layer BEM based forward model from montage and anatomy"""
        try:
            montage = self.montage
            subjects_dir = self.subjects_dir
            subject = self.subject
            spacing = self.spacing
            conductivity = self.conductivity
            trans_file = self.trans_file
            dest_dir = self.dest_dir
            n_jobs = self.n_jobs
            verbose = self.verbose

            try:
                montage = mne.channels.read_montage(kind=montage)
            except Exception:
                raise BadInputFile('Bad montage file: {}'.format(montage))

            fiducials = ['LPA', 'RPA', 'Nz', 'FidT9', 'FidT10', 'FidNz']
            logging.info('SUBJECTS_DIR is set to {}'.format(subjects_dir))
            logging.info('Spacing is set to {}'.format(spacing))
            logging.info('Setting up the source space ...')
            src = mne.setup_source_space(subject, spacing=spacing,
                                         subjects_dir=subjects_dir,
                                         add_dist=False, verbose=verbose)
            self.progress.emit(25)

            logging.info('Creating bem model (be patient) ...')
            model = mne.make_bem_model(subject=subject, ico=4,
                                       conductivity=conductivity,
                                       subjects_dir=subjects_dir,
                                       verbose=verbose)
            # raise Exception('Some catastrophic shit happened')
            self.progress.emit(50)
            bem = mne.make_bem_solution(model, verbose=verbose)
            self.progress.emit(75)
            if not trans_file:
                trans_file = None
            n_jobs = n_jobs
            logging.info('Computing forward solution (be patient) ...')
            ch_names = montage.ch_names
            ch_names = [c for c in ch_names if c not in fiducials]
            info = mne.create_info(ch_names, sfreq=1, ch_types='eeg')
            raw = mne.io.RawArray(np.ones([len(info['ch_names']), 1]), info)
            raw.set_montage(montage)

            ch_names = montage.ch_names
            ch_names = [c for c in ch_names if c not in fiducials]
            fwd = mne.make_forward_solution(raw.info, trans=trans_file,
                                            src=src, bem=bem, meg=False,
                                            eeg=True, mindist=5.0,
                                            n_jobs=n_jobs, verbose=verbose)
            self.progress.emit(100)

            fwd_name = '-'.join(
                [self.subject, 'eeg', spacing, montage.kind, 'fwd.fif'])
            self.fwd_savename = op.join(dest_dir, montage.kind,
                                        spacing, fwd_name)
            mne.write_forward_solution(self.fwd_savename, fwd, overwrite=True)
        except Exception as exc:
            self.exception_ocurred.emit(exc)


class ForwardSetupDialog(QDialog):
    """Dialog window for anatomy and forward model setup"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_ok_to_close = True

        # -------- create widgets -------- #
        self.anatomy_gpbox = AnatomyGroupbox()
        self.forward_gpbox = ForwardOptionsRadioButtons()
        self.fwd_geom_gpbox = FwdGeomGroupbox(self.anatomy_gpbox.default_subj)
        self.compute_fwd_gpbox = ComputeForwardGroupbox()
        self.import_fwd_gpbox = ImportForwardGroupbox()
        # -------------------------------- #

        # -------- setup widgets and connects slots -------- #
        self.compute_fwd_gpbox.is_active = False
        self.import_fwd_gpbox.is_active = False

        self.anatomy_gpbox.use_anatomy_custom_radio.toggled.connect(
            self._on_anatomy_radio_toggled)

        self.forward_gpbox.use_default_fwd_radio.toggled.connect(
            self._on_fwd_option_toggled)

        self.forward_gpbox.import_fwd_radio.toggled.connect(
            self._on_fwd_option_toggled)

        self.forward_gpbox.compute_fwd_radio.toggled.connect(
            self._on_fwd_option_toggled)

        self.anatomy_gpbox.state_changed.connect(
            self._states_changed)
        self.compute_fwd_gpbox.state_changed.connect(
            self._states_changed)
        self.import_fwd_gpbox.state_changed.connect(
            self._states_changed)
        self.fwd_geom_gpbox.state_changed.connect(
            self._states_changed)
        # -------------------------------------------------- #

        # ------------- layout ------------- #
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.anatomy_gpbox)
        main_layout.addWidget(self.forward_gpbox)
        main_layout.addWidget(self.fwd_geom_gpbox)
        main_layout.addWidget(self.import_fwd_gpbox)
        main_layout.addWidget(self.compute_fwd_gpbox)

        outer_layout = QVBoxLayout()
        self.dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok |
                                               QDialogButtonBox.Cancel)
        self.dialog_buttons.accepted.connect(self._on_ok)
        self.dialog_buttons.rejected.connect(self.reject)
        outer_layout.addLayout(main_layout)
        outer_layout.addWidget(self.dialog_buttons)
        self.setLayout(outer_layout)
        # ---------------------------------- #

        self.subjects_dir = self.anatomy_gpbox.subjects_dir
        self.subject = self.anatomy_gpbox.subject

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.min_window_size = self.window().size()

        self.logger = logging.getLogger(type(self).__name__)

    def _on_anatomy_radio_toggled(self):
        if self.anatomy_gpbox.use_anatomy_custom_radio.isChecked():
            if self.forward_gpbox.use_default_fwd_radio.isChecked():
                self.forward_gpbox.compute_fwd_radio.setChecked(True)
            self.forward_gpbox.use_default_fwd_radio.setDisabled(True)
        else:
            self.forward_gpbox.use_default_fwd_radio.setDisabled(False)

    def _on_fwd_option_toggled(self):
        if self.forward_gpbox.use_default_fwd_radio.isChecked():
            self.import_fwd_gpbox.is_active = False
            self.compute_fwd_gpbox.is_active = False
            self.fwd_geom_gpbox.is_active = True
            self.fwd_geom_gpbox.is_use_available = True
        elif self.forward_gpbox.import_fwd_radio.isChecked():
            self.import_fwd_gpbox.is_active = True
            self.compute_fwd_gpbox.is_active = False
            self.fwd_geom_gpbox.is_active = False
        elif self.forward_gpbox.compute_fwd_radio.isChecked():
            self.import_fwd_gpbox.is_active = False
            self.compute_fwd_gpbox.is_active = True
            self.fwd_geom_gpbox.is_active = True
            self.fwd_geom_gpbox.is_use_available = False
        QTimer.singleShot(0, self._fixSize)

    def _fixSize(self):
        """Fix widget size after some of subwidgets were hidden"""
        size = self.sizeHint()
        self.resize(size)

    def _states_changed(self):
        anat_is_good = self.anatomy_gpbox.is_good
        comp_is_good = self.compute_fwd_gpbox.is_good
        imp_is_good = self.import_fwd_gpbox.is_good
        geom_is_good = self.fwd_geom_gpbox.is_good
        if comp_is_good and imp_is_good and geom_is_good and anat_is_good:
            self.dialog_buttons.button(QDialogButtonBox.Ok).setDisabled(False)
        else:
            self.dialog_buttons.button(QDialogButtonBox.Ok).setDisabled(True)

    def _on_ok(self):
        self.is_ok_to_close = True
        self.subjects_dir = self.anatomy_gpbox.subjects_dir
        self.subject = self.anatomy_gpbox.subject
        self.default_fwds_path = self.fwd_geom_gpbox.default_forwards_path
        if self.forward_gpbox.use_default_fwd_radio.isChecked():
            # montage = self.fwd_geom_gpbox.montages_combo.currentText()
            montage = self.fwd_geom_gpbox.montage
            spacing = self.fwd_geom_gpbox.spacings_combo.currentText()
            fwd_name = self.fwd_geom_gpbox.forwards_combo.currentText()
            fwd_folder = op.join(self.default_fwds_path, montage, spacing)
            self.fwd_path = op.join(fwd_folder, fwd_name)

        elif self.forward_gpbox.import_fwd_radio.isChecked():
            path = self.import_fwd_gpbox.select_fwd_dialog.path_ledit.text()
            if op.isfile(path):
                try:
                    self.fwd_path = self.copy_forward_to_folders_struct(path)
                except Exception as e:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText(
                        'Failed to copy {} inside'
                        ' cognigraph folders structure'.format(path))
                    msg.setDetailedText(str(e))
                    msg.show()
            else:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setText('{} is not a file')

        elif self.forward_gpbox.compute_fwd_radio.isChecked():
            # -------------------- gather parameters -------------------- #
            compute_gpbox = self.compute_fwd_gpbox
            geom_gpbox = self.fwd_geom_gpbox
            conductivity = (
                compute_gpbox.brain_conductivity_spinbox.value(),
                compute_gpbox.skull_conductivity_spinbox.value(),
                compute_gpbox.scalp_conductivity_spinbox.value())

            montage = geom_gpbox.montage
            spacing = geom_gpbox.spacings_combo.currentText()
            trans_file = compute_gpbox.select_coreg_widget.path_ledit.text()
            # dest_dir = op.join(self.default_fwds_path, montage, spacing)
            dest_dir = self.default_fwds_path
            # ----------------------------------------------------------- #

            thread_run = ComputeForwardInThread(montage, self.subjects_dir,
                                                self.subject, spacing,
                                                conductivity, trans_file,
                                                dest_dir, n_jobs=1,
                                                verbose=True, parent=self)
            self.is_ok_to_close = thread_run.no_blocking_execution()
            if self.is_ok_to_close:
                self.fwd_path = thread_run.fwd_savename

        if self.is_ok_to_close:
            print(self.fwd_path, self.subject, self.subjects_dir)
            self.accept()

    def copy_forward_to_folders_struct(self, src_path,
                                       montage='imported',
                                       spacing='imported'):
        dest_dir = op.join(self.default_fwds_path, montage, spacing)
        fname = op.split(src_path)[1]
        dest_path = op.join(dest_dir, fname)
        try:
            os.makedirs(dest_dir)
        except OSError:
            pass
        logger.info('Copying to {}'.format(dest_dir))
        copyfile(src_path, dest_path)
        return dest_path


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dialog = ForwardSetupDialog()
    dialog.show()
    sys.exit(app.exec_())
