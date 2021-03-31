#%% md

<img style="float: right;" src="figures/ARAMIS_logo.png">

## My first EEG dataset
#### Marie-Constance Corsi, Aramis lab, Paris

#%% md

## Overview

* Generalities
* EEG Visualization
* Preprocessing
* ERD/S Analysis
* Source reconstruction
* Concluding remarks / FAQs

#%% md

<img style="float: right;" src="figures/python.png" alt="drawing" width="120">

### Generalities

This demo relies on several open-source Python packages:

* [MNE-Python](https://mne.tools/stable/index.html) & [[Gramfort et al, 2014]](https://pubmed.ncbi.nlm.nih.gov/24161808/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org/stable/index.html)
* [Seaborn](https://seaborn.pydata.org)
<img style="float: right;" src="figures/mne.png" alt="drawing" width="120">


The packages used for running this notebook are indicated in the requirements.txt file and could be installed with ``` pip install -r requirements.txt ```

#### Dataset
Use of an open dataset from TODO: XXXX

#### Code
Use of a code adapted from [a mne-python tutorial](https://mne.tools/stable/auto_tutorials/intro/plot_10_overview.html#sphx-glr-auto-tutorials-intro-plot-10-overview-py)

#### /!\ Warning /!\
This demo aims at providing an overview of the main methods used to analyze M/EEG data.
The choice and the parametrization of the methods may differ depending on your dataset and your hypothesis !

#%%

#Import the packages that will be used:
%matplotlib widget
import ipympl
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#%% md

#### Step 0: Load dataset & plot raw data

#%%

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

#%%

print(raw)
print(raw.info)

#%%

raw.plot_psd(fmax=50)
raw.plot(duration=5, n_channels=30)

#%% md

#### Step 1:  Preprocessing

#%%

# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)

#%%

orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

# show some frontal channels to clearly illustrate the artifact removal
chs = ['MEG 0111', 'MEG 0121', 'MEG 0131', 'MEG 0211', 'MEG 0221', 'MEG 0231',
       'MEG 0311', 'MEG 0321', 'MEG 0331', 'MEG 1511', 'MEG 1521', 'MEG 1531',
       'EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006',
       'EEG 007', 'EEG 008']
chan_idxs = [raw.ch_names.index(ch) for ch in chs]
orig_raw.plot(order=chan_idxs, start=12, duration=4)
raw.plot(order=chan_idxs, start=12, duration=4)

#%% md

#### Step 2: Detecting experimental events

#%%

events = mne.find_events(raw, stim_channel='STI 014')
print(events[:5])  # show the first 5

#%%

# create a dictionary associated with the conditions
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}

fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)

#%% md

#### Step 3: Epoching continuous data

#%%

reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV

epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)

#%%

conds_we_care_about = ['auditory/left', 'auditory/right',
                       'visual/left', 'visual/right']
epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
aud_epochs = epochs['auditory']
vis_epochs = epochs['visual']
del raw, epochs  # free up memory

#%%

aud_epochs.plot_image(picks=['MEG 1332', 'EEG 021'])

#%% md

#### Step 4: Time-frequency analysis

#%%

frequencies = np.arange(7, 30, 3)
power = mne.time_frequency.tfr_morlet(aud_epochs, n_cycles=2, return_itc=False,
                                      freqs=frequencies, decim=3)
power.plot(['MEG 1332'])

#%% md

#### Step 5: Estimating evoked responses

#%%

aud_evoked = aud_epochs.average()
vis_evoked = vis_epochs.average()

mne.viz.plot_compare_evokeds(dict(auditory=aud_evoked, visual=vis_evoked),
                             legend='upper left', show_sensors='upper right')

#%%

aud_evoked.plot_joint(picks='eeg')
aud_evoked.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2], ch_type='eeg')

#%%

evoked_diff = mne.combine_evoked([aud_evoked, vis_evoked], weights=[1, -1])
evoked_diff.pick_types(meg='mag').plot_topo(color='r', legend=False)

#%% md

#### Step 6: Inverse modelling

#%%

# load inverse operator
inverse_operator_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                     'sample_audvis-meg-oct-6-meg-inv.fif')
inv_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_file)
# set signal-to-noise ratio (SNR) to compute regularization parameter (λ²)
snr = 3.
lambda2 = 1. / snr ** 2
# generate the source time course (STC)
stc = mne.minimum_norm.apply_inverse(vis_evoked, inv_operator,
                                     lambda2=lambda2,
                                     method='MNE')  # or dSPM, sLORETA, eLORETA

#%%

# path to subjects' MRI files
subjects_dir = os.path.join(sample_data_folder, 'subjects')
# plot
stc.plot(initial_time=0.1, hemi='split', views=['lat', 'med'],
         subjects_dir=subjects_dir)

#%%


