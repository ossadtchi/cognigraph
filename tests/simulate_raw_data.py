"""
===========================
Generate simulated raw data
===========================

This example generates raw data by repeating a desired source
activation multiple times.

"""
# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import read_source_spaces, find_events, Epochs, compute_covariance
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw

print(__doc__)

# data_path = sample.data_path()
data_path = '/home/dmalt/Data/cognigraph/data/'

raw_fname = data_path + 'Koleno.fif'

trans_fname = data_path + 'sample_aligned-trans.fif'
src_fname = data_path + 'sample-oct-6-src.fif'
bem_fname = data_path + 'sample-bem-sol.fif'

# sample_path = '/home/dmalt/mne_data/MNE-sample-data/MEG/sample/labels/'
sample_path = '/home/dmalt/mne_data/MNE-sample-data/subjects/sample/label/'
fname_label = [sample_path + '/lh.MT.label',
               sample_path + '/rh.MT.label']

# fname_label = [sample_path + '/Vis-rh.label',
#                sample_path + '/Vis-lh.label']

# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]

# Load real data as the template
raw = mne.io.read_raw_fif(raw_fname)
raw = raw.crop(0., 30.)  # 30 sec is enough

##############################################################################
# Generate dipole time series
n_dipoles = 2  # number of dipoles to create
epoch_duration = 2.  # duration of each epoch/event
n = 0  # harmonic number


def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of 10Hz"""
    global n
    n_samp = len(times)
    window = np.zeros(n_samp)
    start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
                   for ii in (2 * n, 2 * n + 1)]
    window[start:stop] = 1.
    n += 1
    data = 2e-7 * np.sin(2. * np.pi * 10.  * times)
    data *= window
    return data

times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
src = read_source_spaces(src_fname)
stc_sim = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times, labels=labels,
                          data_fun=data_fun, random_state=0)

# look at our source data
fig, ax = plt.subplots(1)
ax.plot(times, 1e9 * stc_sim.data.T)
ax.set(ylabel='Amplitude (nAm)', xlabel='Time (sec)')
fig.show()

##############################################################################
# Simulate raw data
raw_sim = simulate_raw(raw, stc_sim, trans_fname, src, bem_fname, cov='simple',
                       iir_filter=[0.2, -0.2, 0.04], ecg=False, blink=False,
                       n_jobs=1, verbose=True)
raw_sim.plot()

##############################################################################
# Plot evoked data
events = find_events(raw_sim)  # only 1 pos, so event number == 1
epochs = Epochs(raw_sim, events, 1, -0.2, epoch_duration)
cov = compute_covariance(epochs, tmax=0., method='empirical')  # quick calc
evoked = epochs.average()
evoked.plot_white(cov)
