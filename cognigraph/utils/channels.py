import os.path as op
import mne
from mne.io.meas_info import _kind_dict
import numpy as np
import logging
from .. import MISC_CHANNEL_TYPE

logger = logging.getLogger(__name__)


def fill_eeg_channel_locations(info: mne.Info):
    """
    Add standard locations for EEG channels if none were provided.
    Operates in place.
    """

    # To interpolate we need all EEG channels to have locations, including bad
    # channels
    eeg_channel_ids = mne.pick_types(info, eeg=True, exclude=[])

    # EEG electrode locations are stored in the first three elements of the
    # 'loc' list in info['chs'][idx].  If no location is set, then all of them
    # will be zero.
    channels_wo_location = [
        info["chs"][eeg_channel_id]
        for eeg_channel_id in eeg_channel_ids
        if np.all(info["chs"][eeg_channel_id]["loc"][:3] == [0, 0, 0])
    ]

    if any(channels_wo_location):
        logger.warning(
            "{ch_count} EEG channels have no locations assigned."
            " Default values will be used.".format(
                ch_count=len(channels_wo_location)
            )
        )
        eeg_montage = mne.channels.read_montage(kind="standard_1005")
        for channel in channels_wo_location:
            try:
                channel_idx_in_montage = eeg_montage.ch_names.index(
                    channel["ch_name"]
                )
            except ValueError:
                ch_name = channel["ch_name"]
                logger.warning(
                    "Could not find channel {name} in the standard montage."
                    " Will set it as {misc}".format(
                        name=ch_name, misc=MISC_CHANNEL_TYPE
                    )
                )
                _set_channel_as_misc(channel)
            else:
                channel["loc"][:3] = eeg_montage.pos[channel_idx_in_montage]


def _set_channel_as_misc(channel):
    kind = _kind_dict[MISC_CHANNEL_TYPE][0]
    channel["kind"] = kind


def read_channel_types(info: mne.Info):
    return [mne.io.pick.channel_type(info, i) for i in range(info["nchan"])]


def channel_labels_saver(mne_info: mne.Info):
    return tuple(mne_info["ch_names"]), tuple(mne_info["bads"])


def get_average_reference_projection(channel_count: int):
    """
    Calculates average-reference projection matrix assuming
    all channel_count channels are used

    Parameters
    ----------
    channel_count: int
        number of channels

    Returns
    ------
    np.ndarray
        Projection matrix of shape (channel_count, channel_count)

    """
    n = channel_count
    return np.eye(n) - np.ones((n, n)) / n


def save_montage(montage, save_dir, save_units="mm", overwrite=False):
    """Save eeg montage in .elc format"""
    name = montage.kind
    ext = ".elc"
    save_path = op.join(save_dir, name + ext)

    header = "# ASA electrode file\n"
    reference = "ReferenceLabel\tavg\n"
    units_str = "UnitPosition\t" + save_units + "\n"
    number_pos_str = "NumberPositions=\t%s\n" % len(montage.ch_names)
    if montage.pos.max() > 1:
        orig_units = "mm"
    else:
        orig_units = "m"

    if save_units == "mm" and orig_units == "m":
        scale = 1e3
    elif save_units == "m" and orig_units == "mm":
        scale = 1e-3
    else:
        scale = 1

    fid_mapping = dict(FidT9="LPA", FidT10="RPA", FidNz="Nz")
    for i, ch_name in enumerate(montage.ch_names):
        if ch_name in fid_mapping:
            montage.ch_names[i] = fid_mapping[ch_name]

    if not op.isfile(save_path) or overwrite is True:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(reference)
            f.write(units_str)
            f.write(number_pos_str)
            f.write("Positions\n")
            for p in montage.pos:
                f.write(" ".join([str(c * scale) for c in p]) + "\n")
            f.write("Labels\n")
            for c in montage.ch_names:
                f.write(c + "\n")
    else:
        raise FileExistsError("Montage file '%s' already exists!" % save_path)
    return save_path


def capitalize_chnames(info):
    """Convert channel names in info to upper case; operates inplace"""
    for i, c in enumerate(info['ch_names']):
        info['ch_names'][i] = c.upper()
        info['chs'][i]['ch_name'] = c.upper()


def capitalize_chnames_fwd(fwd):
    capitalize_chnames(fwd['info'])
    for i, c in enumerate(fwd['sol']['row_names']):
        fwd['sol']['row_names'][i] = c.upper()
