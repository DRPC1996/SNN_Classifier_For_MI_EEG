
#%%
import numpy as np
import scipy.io 

m = scipy.io.loadmat('BCICIV_1_mat\BCICIV_calib_ds1d.mat', struct_as_record=True)

#%%
sample_rate = m['nfo']['fs'][0][0][0][0]
EEG = m['cnt'].T
nchannels, nsamples = EEG.shape

channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
event_onsets = m['mrk'][0][0][0]
event_codes = m['mrk'][0][0][1]

labels = np.zeros((1,nsamples), int)
labels[0, event_onsets] = event_codes

cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
cl1 = cl_lab[0]
cl2 = cl_lab[1]

nclasses = len(cl_lab)
nevents = len(event_onsets)

#%%
trials = {}

win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

nsamples = len(win)

#%%
for cl, code in zip(cl_lab, np.unique(event_codes)):

    cl_onsets = event_onsets[event_codes == code]

    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

    for i, onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEG[:, win+onset]

#%%
from matplotlib import mlab

def psd(trials):

    ntrials =  trials.shape[2]
    trials_PSD = np.zeros((nchannels,101,ntrials))

    for trial in range(ntrials):
        for ch in range(nchannels):
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
            trials_PSD[ch, :, trial] = PSD.ravel()
    
    return trials_PSD, freqs 

#%%
psd_r, freqs = psd(trials[cl1])
psd_f, freqs = psd(trials[cl2])
trials_PSD = {cl1: psd_r, cl2:psd_f}

#%%
import matplotlib.pyplot as plt

def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    
    plt.figure(figsize=(12,5))

    nchans = len(chan_ind)

    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)

    for i,ch in enumerate(chan_ind):
        plt.subplot(nrows,ncols,i+1)

        for cl in trials.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)

            plt.xlim(1,30)

            if maxy != None:
                plt.ylim(0,maxy)

            plt.grid()

            plt.xlabel('Frequency (Hz)')

            if chan_lab == None:
                plt.title('Channel %d' % (ch+1))
            else:
                plt.title(chan_lab[i])

            plt.legend()

        plt.tight_layout()

# %%
plot_psd(
    trials_PSD,
    freqs,
    [channel_names.index(ch) for ch in ['C3','Cz','C4']],
    chan_lab=['left','center', 'right'],
    maxy=500
)

# %%
import scipy.signal

def bandpass(trials, lo, hi, sample_rate):

    a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a,b, trials[:,:,i], axis=1)

    return trials_filt
# %%

trials_filt = {cl1: bandpass(trials[cl1],8,15,sample_rate),
                cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
# %%

psd_r, freqs = psd(trials_filt[cl1])
psd_f, freqs = psd(trials_filt[cl2])
trials_PSD = {cl1: psd_r, cl2:psd_f}

# %%
plot_psd(
    trials_PSD,
    freqs,
    [channel_names.index(ch) for ch in ['C3','Cz','C4']],
    chan_lab=['left','center', 'right'],
    maxy=500
)
#%%
def logvar(trials):
    return np.log(np.var(trials, axis=1))

#%%
trials_logvar = {cl1: logvar(trials[cl1]),
                cl2: logvar(trials[cl2])}

#%%
def plot_logvar(trials):
     plt.figure(figsize=(12,5))

     x0 = np.arange(nchannels)
     x1 = np.arange(nchannels) + 0.4

     y0 = np.mean(trials[cl1], axis=1)
     y1 = np.mean(trials[cl2], axis=1)

     plt.bar(x0,y0, width=0.5, color='b')
     plt.bar(x1,y1, width=0.4, color='r')

     plt.xlim(-0.5, nchannels+0.5)

     plt.gca().yaxis.grid(True)
     plt.title('long-var of each channel/component')
     plt.xlabel('channels/components')
     plt.ylabel('log-var')
     plt.legend(cl_lab)

#%% 
plot_logvar(trials_logvar)
# %%
from numpy import linalg

def cov(trials):
    ntrials = trials.shape[2]
    covs = [trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    return np.mean(covs, axis=0)

def whitening(sigma):
    U, l, _ = linalg.svd(sigma)
    return U.dot(np.diag(l**-0.5))

def csp(trials_r, trials_f):
    cov_r = cov(trials_r)
    cov_f = cov(trials_f)
    P = whitening(cov_r + cov_f)
    B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P))
    W = P.dot(B)
    return W

def apply_mix(W,trials):
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp

# %%
W = csp(trials_filt[cl1], trials_filt[cl2])
trials_csp = {cl1: apply_mix(W,trials_filt[cl1]),
                cl2: apply_mix(W,trials_filt[cl2])}

#%%
trials_logvar = {cl1: logvar(trials_csp[cl1]),
                cl2: logvar(trials_csp[cl2])}

plot_logvar(trials_logvar)

#%%
psd_r, freqs = psd(trials_csp[cl1])
psd_f, freqs = psd(trials_csp[cl2])
trials_PSD = {cl1: psd_r, cl2:psd_f}

plot_psd(trials_PSD, freqs, [0,29,58], chan_lab=['1st components','middle component','last component'], maxy=0.75)



#%%
def plot_scatter(left, foot):
    plt.figure()
    plt.scatter(left[0,:], left[-1,:], color='b')
    plt.scatter(foot[0,:], foot[-1,:], color='r')
    plt.xlabel('last component')
    plt.ylabel('first component')

plot_scatter(trials_logvar[cl1],trials_logvar[cl2])


# %%
