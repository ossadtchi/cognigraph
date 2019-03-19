"""Dialogs for cognigraph parameters setup
author: dmalt
date: 2019-02-19

"""
import os
import os.path as op
import inspect
import json
import shutil
import numpy as np


import mne
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QComboBox,
                             QDialogButtonBox, QPushButton, QFileDialog,
                             QRadioButton, QGroupBox, QVBoxLayout, QLineEdit,
                             QWidget, QDoubleSpinBox, QMessageBox, QMainWindow)

from PyQt5.Qt import QSizePolicy, QTimer
from PyQt5.QtCore import Qt, pyqtSignal, QSignalBlocker
from cognigraph import COGNIGRAPH_DATA
from cognigraph.gui.async_pipeline_update import ThreadToBeWaitedFor

import logging


class ResettableComboBox(QComboBox):
    """
    Combobox with capability of setting all items
    anew and retrieving items as list.

    """
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
        min_lineedit_width = fm.boundingRect(COGNIGRAPH_DATA).width()
        self.path_ledit.setMinimumWidth(min_lineedit_width)
        self.browse_button = QPushButton('Browse')
        self.browse_button.setDefault(False)
        self.browse_button.setAutoDefault(True)
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
    QLineEdit + QPushButton connected to QFileDialog to set path to FOLDER

    """

    def _on_browse_clicked(self):
        self.path = self.file_dialog.getExistingDirectory()
        super()._on_browse_clicked()


class FileSelectorWidget(PathSelectorWidget):
    """
    QLineEdit + QPushButton connected to QFileDialog to set path to FILE

    """
    def __init__(self, dialog_caption, file_filter, path='', parent=None):
        super().__init__(dialog_caption, path, parent)
        self.filter = file_filter

    def _on_browse_clicked(self):
        self.path = self.file_dialog.getOpenFileName(filter=self.filter)[0]
        super()._on_browse_clicked()


class StateAwareGroupbox(QGroupBox):
    """
    QGroupBox tracking valid-invalid and active-inactive state
    and emitting a signal when these states change.

    """
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


class FwdOptionsRadioButtons(QGroupBox):
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


class AnatGroupbox(StateAwareGroupbox):
    """Groupbox to setup subjects_dir and subject"""
    def __init__(self, title='Anatomy',
                 default_subj_dir=op.join(COGNIGRAPH_DATA, 'anatomy'),
                 default_subj='fsaverage', parent=None):
        super().__init__(parent, title=title)
        self.default_subj_dir = default_subj_dir
        self.default_subj = default_subj
        self.subjects_dir = self.default_subj_dir
        self.subject = self.default_subj
        self.subjects = self._get_fsf_subjects(self.subjects_dir)
        # ------------------ anatomy gpbox ------------------ #
        self.use_avail_anat_radio = QRadioButton('&Use available anatomy')
        self.use_avail_anat_radio.setChecked(True)
        self.use_avail_anat_radio.toggled.connect(self._on_toggled)
        self.import_anat_radio = QRadioButton('&Import anatomy')
        self.import_anat_radio.setChecked(False)

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
            self._on_anat_path_changed)

        self.subject_combobox.currentTextChanged.connect(
            self._on_subject_combo_changed)

        self.anat_path_widget = QWidget()
        self.anat_path_widget.setLayout(subjects_dir_subject_layout)
        self.anat_path_widget.setSizePolicy(QSizePolicy.Minimum,
                                            QSizePolicy.Fixed)
        self.subjects_dir_widget.setDisabled(True)

        anat_gbox_layout = QVBoxLayout()
        anat_gbox_layout.addWidget(self.use_avail_anat_radio)
        anat_gbox_layout.addWidget(self.import_anat_radio)
        anat_gbox_layout.addWidget(self.anat_path_widget)

        self.setLayout(anat_gbox_layout)
        # ------------------------------------------------------ #

    def _on_toggled(self):
        if self.use_avail_anat_radio.isChecked():
            self.subjects_dir_widget.setDisabled(True)
            if self.subjects_dir != self.default_subj_dir:
                self.subjects_dir_widget.path_ledit.setText(
                    self.default_subj_dir)
            self.is_valid = True
        else:
            if self.subjects_dir and self.subject:
                self.is_valid = True
            else:
                self.is_valid = False
            self.subjects_dir_widget.setDisabled(False)
            self.subjects_dir_widget.browse_button.setFocus()

    def _get_fsf_subjects(self, path):
        files = os.listdir(path)
        return sorted([f for f in files if op.isdir(op.join(path, f))
                       and 'surf' in os.listdir(op.join(path, f))])

    def _on_anat_path_changed(self):
        new_path = self.subjects_dir_widget.path_ledit.text()
        if new_path != self.subjects_dir:
            self.subjects = self._get_fsf_subjects(new_path)
            self.subject_combobox.setItems(sorted(self.subjects))
        self.subjects_dir = new_path
        # If valid freesurfer subjects were found:
        if self.subject_combobox.currentText():
            self.subjects_dir = self.subjects_dir
            self.subject = self.subject_combobox.currentText()
            self.is_valid = True
        else:
            self.is_valid = False

    def _on_subject_combo_changed(self):
        self.subject = self.subject_combobox.currentText()


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
            COGNIGRAPH_DATA, 'forwards', default_subj)
        self._get_available_forwards(self.default_forwards_path)
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

        self.use_default_montage_radio = QRadioButton('&Use default')
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
        montages_desc_path = op.join(COGNIGRAPH_DATA, 'montages_desc.json')
        with open(montages_desc_path, 'r') as f:
            description = json.load(f)
            self.montages_desc = description['montages']
            self.spacings_desc = description['spacings']
        self.builtin_montage_names = sorted(
            mne.channels.get_builtin_montages())

    def _get_available_forwards(self, folder):
        p = folder
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
        self.montage = self.default_montage
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
            self.select_montage_dialog.browse_button.setFocus()

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


class ImportFwdGroupbox(StateAwareGroupbox):
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


class ComputeFwdGroupbox(StateAwareGroupbox):
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
            self.select_coreg_widget.browse_button.setFocus()

    def _on_path_changed(self):
        if self.select_coreg_widget.path_ledit.text():
            self.is_valid = True


class BadInputFile(Exception):
    pass


class ComputeFwdInThread(ThreadToBeWaitedFor):
    """Compute forward model in parallel thread"""

    def __init__(self, montage, subjects_dir, subject,
                 spacing, conductivity, trans_file, dest_dir, n_jobs=8,
                 verbose='ERROR', parent=None):
        super().__init__(parent=parent)
        self.progress_text = 'Computing forward model... Please be patient.'
        self.error_text = 'Forward model computation failed.'
        self.montage = montage
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.spacing = spacing
        self.conductivity = conductivity
        self.trans_file = trans_file
        self.dest_dir = dest_dir
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fwd_savename = None
        self.is_show_progress = True

        montage_kind = op.splitext(op.basename(montage))[0]
        fwd_name = '-'.join([subject, 'eeg', spacing, montage_kind, 'fwd.fif'])
        self.fwd_savename = op.join(dest_dir, montage_kind, spacing, fwd_name)

    def no_blocking_execution(self):
        if op.isfile(self.fwd_savename):
            ans = QMessageBox.question(
                self.parent(), 'Destination file exists',
                'Forward model file "{}" already exists.'
                ' Recomtute?'.format(self.fwd_savename),
                QMessageBox.Yes | QMessageBox.No)
            if ans == QMessageBox.Yes:
                return super().no_blocking_execution()
            else:
                return True  # success since forward model is already there
        else:
            return super().no_blocking_execution()

    def _run(self):
        """Compute 3-layer BEM based forward model from montage and anatomy"""
        montage = self.montage
        subjects_dir = self.subjects_dir
        subject = self.subject
        spacing = self.spacing
        conductivity = self.conductivity
        trans_file = self.trans_file
        dest_dir = self.dest_dir
        n_jobs = self.n_jobs
        verbose = self.verbose
        self.logger.debug('Computing forward with the following parameters.')
        self.logger.debug('montage: %s', montage)
        self.logger.debug('SUBJECT: %s', subject)
        self.logger.debug('SUBJECTS_DIR: %s', subjects_dir)
        self.logger.debug('spacing: %s', spacing)
        self.logger.debug('trans_file: %s', trans_file)
        self.logger.debug('conductivity: %s', conductivity)
        self.logger.debug('dest_dir: %s', dest_dir)

        try:
            montage = mne.channels.read_montage(kind=montage)
        except Exception:
            raise BadInputFile('Bad montage file: {}'.format(montage))

        os.makedirs(op.dirname(self.fwd_savename), exist_ok=True)

        fiducials = ['LPA', 'RPA', 'Nz', 'FidT9', 'FidT10', 'FidNz']
        self.logger.info('Setting up the source space ...')
        src = mne.setup_source_space(subject, spacing=spacing,
                                     subjects_dir=subjects_dir,
                                     add_dist=False, verbose=verbose)
        self.progress_updated.emit(25)

        # raise Exception('Some catastrophic shit happened')
        self.logger.info('Creating bem model (be patient) ...')
        model = mne.make_bem_model(subject=subject, ico=4,
                                   conductivity=conductivity,
                                   subjects_dir=subjects_dir,
                                   verbose=verbose)
        self.progress_updated.emit(50)
        bem = mne.make_bem_solution(model, verbose=verbose)
        self.progress_updated.emit(75)
        if not trans_file:
            trans_file = None
        n_jobs = n_jobs
        self.logger.info('Computing forward solution (be patient) ...')
        ch_names = montage.ch_names
        ch_names = [c for c in ch_names if c not in fiducials]
        info = mne.create_info(ch_names, sfreq=1, ch_types='eeg')
        raw = mne.io.RawArray(np.ones([len(info['ch_names']), 1]), info,
                              verbose=verbose)
        raw.set_montage(montage)

        ch_names = montage.ch_names
        ch_names = [c for c in ch_names if c not in fiducials]
        fwd = mne.make_forward_solution(raw.info, trans=trans_file,
                                        src=src, bem=bem, meg=False,
                                        eeg=True, mindist=5.0,
                                        n_jobs=n_jobs, verbose=verbose)
        self.progress_updated.emit(100)

        mne.write_forward_solution(self.fwd_savename,
                                   fwd, overwrite=True, verbose=verbose)


class FwdSetupDialog(QDialog):
    """Dialog window for anatomy and forward model setup"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_ok_to_close = True
        # -------- create widgets -------- #
        self.anat_gpbox = AnatGroupbox()
        self.forward_gpbox = FwdOptionsRadioButtons()
        self.fwd_geom_gpbox = FwdGeomGroupbox(self.anat_gpbox.default_subj)
        self.compute_fwd_gpbox = ComputeFwdGroupbox()
        self.import_fwd_gpbox = ImportFwdGroupbox()
        # -------------------------------- #

        # -------- setup widgets and connects slots -------- #
        self.compute_fwd_gpbox.is_active = False
        self.import_fwd_gpbox.is_active = False

        self.anat_gpbox.import_anat_radio.toggled.connect(
            self._on_anat_radio_toggled)
        self.anat_gpbox.subject_combobox.currentTextChanged.connect(
            self.reset_forward_groupbox)

        self.forward_gpbox.use_default_fwd_radio.toggled.connect(
            self._on_fwd_option_toggled)

        self.forward_gpbox.import_fwd_radio.toggled.connect(
            self._on_fwd_option_toggled)
        self.forward_gpbox.import_fwd_radio.toggled.connect(
            self.import_fwd_gpbox.select_fwd_dialog.browse_button.setFocus)

        self.forward_gpbox.compute_fwd_radio.toggled.connect(
            self._on_fwd_option_toggled)

        self.anat_gpbox.state_changed.connect(
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
        main_layout.addWidget(self.anat_gpbox)
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

        self.subjects_dir = self.anat_gpbox.subjects_dir
        self.subject = self.anat_gpbox.subject

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.min_window_size = self.window().size()

        self.logger = logging.getLogger(type(self).__name__)

    def _on_anat_radio_toggled(self):
        if self.anat_gpbox.import_anat_radio.isChecked():
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
            self.fwd_geom_gpbox._get_available_forwards(
                op.join(
                    COGNIGRAPH_DATA, 'forwards', self.anat_gpbox.subject))
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
        anat_is_good = self.anat_gpbox.is_good
        comp_is_good = self.compute_fwd_gpbox.is_good
        imp_is_good = self.import_fwd_gpbox.is_good
        geom_is_good = self.fwd_geom_gpbox.is_good
        if comp_is_good and imp_is_good and geom_is_good and anat_is_good:
            self.dialog_buttons.button(QDialogButtonBox.Ok).setDisabled(False)
        else:
            self.dialog_buttons.button(QDialogButtonBox.Ok).setDisabled(True)

    def _on_ok(self):
        """Called when OK button is clicked"""
        self.is_ok_to_close = True
        self.subjects_dir = self.anat_gpbox.subjects_dir
        self.logger.debug(
            'Subjects dir is set to "{}".'.format(self.subjects_dir))
        self.subject = self.anat_gpbox.subject
        self.logger.debug('Subject is set to "{}"'.format(self.subject))
        if self.anat_gpbox.import_anat_radio.isChecked():
            self.copy_anat_to_folders_struct(self.subjects_dir, self.subject)
        self.cur_anat_fwds_path = op.join(
            COGNIGRAPH_DATA, 'forwards', self.subject)
        if self.forward_gpbox.use_default_fwd_radio.isChecked():
            # montage = self.fwd_geom_gpbox.montages_combo.currentText()
            montage = self.fwd_geom_gpbox.montage
            spacing = self.fwd_geom_gpbox.spacings_combo.currentText()
            fwd_name = self.fwd_geom_gpbox.forwards_combo.currentText()
            self.fwd_path = op.join(
                self.cur_anat_fwds_path, montage, spacing, fwd_name)

        elif self.forward_gpbox.import_fwd_radio.isChecked():
            path = self.import_fwd_gpbox.select_fwd_dialog.path_ledit.text()
            if op.isfile(path):
                try:
                    self.fwd_path = self.copy_fwd_to_folders_struct(path)
                    # self.fwd_path = path
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
            # dest_dir = op.join(self.cur_anat_fwds_path, montage, spacing)
            dest_dir = self.cur_anat_fwds_path
            # ----------------------------------------------------------- #

            thread_run = ComputeFwdInThread(montage, self.subjects_dir,
                                            self.subject, spacing,
                                            conductivity, trans_file,
                                            dest_dir, n_jobs=1,
                                            verbose='ERROR', parent=self)
            self.is_ok_to_close = thread_run.no_blocking_execution()
            if self.is_ok_to_close:
                self.fwd_path = thread_run.fwd_savename

        if self.is_ok_to_close:
            self.logger.debug(
                'Forward model path is set to "{}"'.format(self.fwd_path))
            self.accept()

    def copy_fwd_to_folders_struct(self, src_path, montage='imported',
                                   spacing='imported'):
        dest_dir = op.join(self.cur_anat_fwds_path, montage, spacing)
        fname = op.split(src_path)[1]
        dest_path = op.join(dest_dir, fname)
        try:
            os.makedirs(dest_dir)
        except OSError:
            self.logger.warning(
                'Destination folder {} exists.'.format(dest_dir))
        self.logger.info('Copying to {}'.format(dest_dir))
        shutil.copyfile(src_path, dest_path)
        return dest_path

    def copy_anat_to_folders_struct(self, src_subjects_dir, subject):
        dest_subjects_dir = op.join(COGNIGRAPH_DATA, 'anatomy')
        src_path = op.join(src_subjects_dir, subject)
        dest_path = op.join(dest_subjects_dir, subject)
        if not op.isdir(op.join(dest_subjects_dir, subject)):
            self.logger.info('Copying anatomy to {}'.format(dest_subjects_dir))
            shutil.copytree(src_path, dest_path)
        elif src_path != dest_path:
            answer = QMessageBox.question(
                self, 'Destination folder exists',
                'Anatomy for subject "{}" exists. Overwrite?'.format(subject),
                QMessageBox.Yes | QMessageBox.No)
            if answer == QMessageBox.Yes:
                self.logger.info(
                    'Overwriting anatomy for subject {}.'.format(subject))
                shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
            else:
                self.logger.info(
                    'Keeping existing anatomy for subject {}.'.format(subject))
        else:
            self.logger.info('Source and destination folders are the same.'
                             ' Skipping.')
        return dest_path

    def reset_forward_groupbox(self):
        if (self.anat_gpbox.use_avail_anat_radio.isChecked() and
                self.forward_gpbox.use_default_fwd_radio.isChecked()):
            self.fwd_geom_gpbox._get_available_forwards(
                op.join(
                    COGNIGRAPH_DATA, 'forwards', self.anat_gpbox.subject))
            self.fwd_geom_gpbox.is_use_available = True


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    format = '%(asctime)s:%(name)-17s:%(levelname)s:%(message)s'
    logging.basicConfig(level=logging.DEBUG, filename=None, format=format)

    class MW(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.button = QPushButton('Push me')
            self.button.clicked.connect(self._on_clicked)
            self.dialog = FwdSetupDialog(parent=self)
            self.setCentralWidget(self.button)

        def _on_clicked(self):
            self.dialog.show()

    app = QApplication(sys.argv)
    # wind = MW()
    # wind.setAttribute(Qt.WA_DeleteOnClose)

    dialog = FwdSetupDialog()
    dialog.show()
    # wind.show()
    # wind.show()
    # app.exec_()

    sys.exit(app.exec_())
