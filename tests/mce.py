"""Minimum current estimate"""
# from load_data import *
# def mce(fwd, n_svd):
import numpy as np
from numpy.linalg import svd
from scipy.linalg import block_diag
from scipy.optimize import linprog
from scipy.signal import lfilter
from sklearn.preprocessing import normalize
import mne
from mne.minimum_norm import apply_inverse_raw, make_inverse_operator
# from mne impo


class MCE:
    input = []
    output = []

    def __init__(self, fwd_opr, noise_cov, n_comp, info):
        # -------- truncated svd for fwd_opr operator -------- #
        leadfield = fwd_opr['sol']['data']
        U, S, V = svd(leadfield)

        Sn = np.zeros([n_comp, V.shape[0]])
        Sn[:n_comp, :n_comp] = np.diag(S[:n_comp])

        self.Un = U[:, :n_comp]
        self.A_non_ori = Sn @ V
        # ---------------------------------------------------- #

        self.mne_inv = make_inverse_operator(info, fwd_opr, noise_cov,
                                             depth=0.8, loose=1, fixed=False)
        self.info = info
        self.Sn = Sn
        self.V = V

    def update(self):
        raw_slice = mne.io.RawArray(
                np.expand_dims(self.input, axis=1), self.info)

        # --------------------- get dipole orientations --------------------- #
        stc_slice = apply_inverse_raw(raw_slice, self.mne_inv,
                                      pick_ori='vector',
                                      method='MNE', lambda2=1)
        # print(stc_slice.shape)
        Q = normalize(stc_slice.data[:, :, 0])  # dipole orientations
        QQ = block_diag(*Q).T                   # matrix with dipole orientats
        # ------------------------------------------------------------------- #

        # -------- setup linprog params -------- #
        A_eq = self.A_non_ori @ QQ
        data_slice = raw_c.get_data()[:, slice_ind]
        print('----------------------------------------')
        print(slice_ind)
        b_eq = self.Un.T @ data_slice
        c = np.ones(A_eq.shape[1])
        # -------------------------------------- #

        sol = linprog(c, A_eq=A_eq, b_eq=b_eq,
                      method='interior-point', bounds=(0, None),
                      options={'disp': True})
        self.output = sol.x
        return Q, QQ, A_eq, data_slice, b_eq, c


if __name__ == '__main__':
    # ------------------- setup paths ------------------- #
    h_dir = '/home/dmalt'
    test_data_path = '/home/dmalt/Data/cognigraph/data/'
    fwd_path = h_dir + '/Code/python/playground_cognigraph/'
    fwd_fname = fwd_path + 'dmalt_custom-fwd.fif'
    subjects_dir = '/home/dmalt/mne_data/MNE-sample-data/subjects'
    # --------------------------------------------------- #

    # ----------------- load and filter raw data ----------------- #
    raw = mne.io.Raw(test_data_path + 'raw_sim.fif', preload=True)
    raw.set_eeg_reference(ref_channels='average')
    raw.apply_proj()
    raw.pick_types(eeg=True, stim=False)
    raw_c = raw.copy()
    raw_c.filter(l_freq=8, h_freq=12)
    raw_c.set_eeg_reference(ref_channels='average')
    raw_c.apply_proj()
    # ------------------------------------------------------------ #

    # -------------------- load forward solution -------------------- #
    fwd_ = mne.read_forward_solution(fwd_fname)
    fwd_fix_ = mne.convert_forward_solution(fwd_, surf_ori=True,
                                            force_fixed=False)
    leadfield_ = fwd_fix_['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield_.shape)
    # --------------------------------------------------------------- #

    N_COMPONENTS = 40

    # -------- leadfield dims -------- #
    N_SRC = fwd_fix_['nsource']
    N_SEN = leadfield_.shape[0]
    # -------------------------------- #

    # ------------------------ noise-covariance ------------------------ #
    cov_data = np.identity(N_SEN)
    n_cov = mne.Covariance(cov_data, raw.info['ch_names'],
                           raw.info['bads'], raw.info['projs'], nfree=1)
    # ------------------------------------------------------------------ #

    inverse_operator_ = make_inverse_operator(
            raw.info, fwd_, n_cov, depth=0.8, loose=1, fixed=False)

    mce = MCE(fwd_fix_, n_cov, N_COMPONENTS, raw_c.info)
    # Sn = S[:N_COMPONENTS]

    # data = raw_c.get_data(start=start, stop=stop)

    T_START, T_STOP = 0, 0.5
    T_STEP = 1 / raw_c.info['sfreq']
    start, stop = raw.time_as_index([T_START, T_STOP])
    TIMES = np.arange(T_START, T_STOP, T_STEP)
    # TIMES=TIMES[:13]
    data_mce = np.empty([N_SRC, len(TIMES)])

    INV_METHOD = 'MNE'
    for i, time in enumerate(TIMES):
        slice_ind = raw_c.time_as_index(time)[0]
        mce.input = raw_c.get_data()[:, slice_ind]
        Q_, QQ_, A_eq_, data_slice_, b_eq_, c_ = mce.update()
        data_mce[:, i] = mce.output
    # data_mce[:,0] = sol.x
    # -------------------- setup dummy src space obj -------------------- #
    stc_mce_scal = apply_inverse_raw(
            raw_c, inverse_operator_, method=INV_METHOD,
            lambda2=1, start=start, stop=stop)
    # ------------------------------------------------------------------- #
    stc_mce = stc_mce_scal.copy()

    factor = 0.99
    a = [1, -factor]
    b = [1 - factor]

    smooth = lfilter(b, a, np.abs(data_mce), axis=1)
    stc_mce.data = smooth
    # stc_mce.data = data_mce

    brain = stc_mce.plot(hemi='split', initial_time=0, time_viewer=True,
                         clim=dict(kind='value', lims=[1.e-8, 1.5e-8, 2e-8]),
                         subjects_dir=subjects_dir, transparent=True,  colormap='bwr')

    brain.add_foci(10188, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6)
    brain.show_view('lateral')
    # sol = linprog(c, options={'disp': True})
