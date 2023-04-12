# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:38:21 2023

@author: dsplab
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
from araus_utils import make_logmel_spectrograms
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

import sklearn, os, wget, hashlib, librosa

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

from zipfile import ZipFile

class ARAUS_Sequence_from_npy2(Sequence):
   
    def __init__(self, responses,
                       npy_dir = os.path.join('..','features2'),
                       batch_size = 32,
                       shuffle = True,
                       seed_val = 2021,
                       verbose = 1):
        
        self.responses = responses
        self.n_samples = len(responses)
        self.npy_dir = npy_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed_val = seed_val
        np.random.seed(self.seed_val) # Seed random state based on seed_val
        self.verbose = verbose
        self.order = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)
    
    def __getitem__(self, idx):
      
        '''
        Returns the idx-th batch of samples.
        '''
        # GET DATAFRAME ROWS CORRESPONDING TO CURRENT BATCH OF SAMPLES
        batch_idxs = self.order[idx*self.batch_size : min((idx+1)*self.batch_size, self.n_samples)]
        batch_responses = self.responses.iloc[batch_idxs,:]
        
        # LOAD CURRENT BATCH OF FEATURES
        augmented_spectrograms = []
        alist=[]
        attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
        # Dataframe's index may be out of order so we don't use it (and assign it to _ instead).
        for idx, (_, row) in enumerate(batch_responses.iterrows()): 
            # GET NECESSARY DATA FROM CURRENT ROW
            participant_id = row['participant']
            fold = row['fold_r']
            soundscape_fname = row['soundscape']
            masker_fname = row['masker']
            smr = row['smr']
            stimulus_index = row['stimulus_index']
            att_id = row[attributes].values
            att_id = (att_id-3)/2
            alist.append(att_id)

            # READ FROM EXISTING NPY FILE
            in_fname = f'fold_{fold}_participant_{participant_id:>05}_stimulus_{stimulus_index:02d}.npy'
            in_fpath = os.path.join(self.npy_dir, in_fname)
            
            augmented_spectrograms.append(np.load(in_fpath))
        augmented_spectrograms = np.array(augmented_spectrograms)
 # Define attributes to extract from dataframes
        attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous'] 
# Define weights for each attribute in attributes in computation of ISO Pleasantness
        ISOPl_weights = [1,0,-np.sqrt(2)/2,np.sqrt(2)/2, 0, np.sqrt(2)/2,-1,-np.sqrt(2)/2] 
        ISOPls = ((batch_responses[attributes] * ISOPl_weights).sum(axis=1)/(4+np.sqrt(32))).values
        alist = np.array(alist).astype(np.float32)
        labels = augmented_spectrograms, alist#[spec, nparray of 8 attri]
        return labels # Returns a batch of (inputs, labels) = (augmented_spectrograms, ISO_Pls)
    
    def __len__(self):
        '''
        Returns number of batches in the Sequence (i.e. number
        of batches per epoch).
        '''
      
        return self.n_samples // self.batch_size + (self.n_samples % self.batch_size > 0)
    
    def precompute(self, soundscapes, maskers,
                   overwrite = False,
                   make_augmented_soundscapes_kwargs = {'mode': 'return',
                                                        'verbose': 0},
                   make_logmel_spectrograms_kwargs = {'n_fft': 4096,
                                                      'hop_length': 2048,
                                                      'n_mels': 64,
                                                      'verbose': 0}):
       
        # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
        if not os.path.exists(self.npy_dir): os.makedirs(self.npy_dir)
        
        for idx, (_, row) in enumerate(self.responses.iterrows()): # Dataframe's index may be out of order so we don't use it (and assign it to _ instead).
            if self.verbose > 0: print(f'Progress: {idx+1}/{self.n_samples}.', end = '\r')
            
            # GET NECESSARY DATA FROM CURRENT ROW
            participant_id = row['participant']
            fold = row['fold_r']
            soundscape_fname = row['soundscape']
            masker_fname = row['masker']
            smr = row['smr']
            stimulus_index = row['stimulus_index']

            # CHECK IF FILE EXISTS
            out_fname = f'fold_{fold}_participant_{participant_id:>05}_stimulus_{stimulus_index:02d}.npy'
            out_fpath = os.path.join(self.npy_dir, out_fname)
            if os.path.exists(out_fpath) == False:
                pass
            if os.path.exists(out_fpath) and (not overwrite):
                if self.verbose > 1: print(f'Warning: {out_fpath} already exists, skipping its generation...') 
                continue # Skip all processing steps to save time.
            
            # MAKE SPECTROGRAMS
            _, augmented_soundscape = make_augmented_soundscapes3(row.to_frame().T, soundscapes, maskers,
                                                                 **make_augmented_soundscapes_kwargs)
            logmel_spectrograms = make_logmel_spectrograms(input_data = np.squeeze(augmented_soundscape).T,
                                                           **make_logmel_spectrograms_kwargs).transpose([1,0,2])
            
            # WRITE SPECTROGRAMS TO FILE
            if (type(np.save(out_fpath,logmel_spectrograms)) == type(None)):
                pass
            else:
                np.save(out_fpath,logmel_spectrograms).astype(np.float32)
    def on_epoch_end(self):
        '''
        Method called at the end of every epoch.
        Shuffles samples for next batch if self.shuffle is True.
        '''
        if self.shuffle:
            self.order = np.random.permutation(self.n_samples)
            
#filepath = 'soundscapes/R0012_segment_binaural_44100_2.wav'#request.form['aud']
#flist = filepath.rsplit('/', 2) 
#filename = flist[-1]    
#folder = flist[-2]
            


#create spec for the combined soundscape and masker