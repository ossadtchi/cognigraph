
fname = '/home/dmalt/Data/cognigraph/data/Koleno.vhdr'

# --------- prepare channels ------------ #
ch_path  = '/home/dmalt/Data/cognigraph/channel_BrainProducts_ActiCap_128.mat'
ch_struct = loadmat(ch_path)
kind = ch_struct['Comment'][0]
chans = ch_struct['Channel'][0]
ch_locs = np.empty([len(chans), 3])
ch_types = [None] * len(chans)
ch_names = [None] * len(chans)
selection = np.arange(len(chans))

for i_ch, chan in enumerate(chans):
    ch_names[i_ch] = chan[0][0]
    ch_types[i_ch] = chan[2][0]
    ch_locs[i_ch] = chan[4][:, 0]


ch_locs[:,0:2] = ch_locs[:,-2:-4:-1]
ch_locs[:,0] = -ch_locs[:,0]
# ch_locs = ch_locs * 2
# ch_names[ch_names.index('OI1h')] = 'Ol1h'
# ch_names[ch_names.index('OI2h')] = 'Ol2h'
ch_names[ch_names.index('GND')] = 'AFz'
ch_names[ch_names.index('REF')] = 'FCz'
# ch_names[ch_names.index('TPP9h')] = 'TTP9h'

montage = Montage(ch_locs, ch_names, kind, selection)
# montage.plot()

raw = Raw(fname, preload=True)
raw.set_montage(montage)
raw.info['bads'] = ['F5', 'PPO10h', 'C5', 'FCC2h', 'F2', 'VEOG']

raw_c = raw.copy()
