# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:14:26 2023

@author: Rachel
"""

# https://youtu.be/bluclMxiUkA
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 
Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.
"""
import numpy as np
from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
from araus_utils import make_logmel_spectrograms
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import math
import sklearn, os, wget, hashlib, librosa

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

from zipfile import ZipFile

from flask import Flask, render_template, send_file, make_response, url_for, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import soundfile as sf
#Pandas and Matplotlib
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import plotly.express as px
import matplotlib
import wave
import matplotlib.pyplot as plt
#other requirements
import io
from base64 import b64encode
from io import BytesIO

import glob
import numpy as np
from flask import Flask, request, render_template

import os
import pandas as pd
from araus_utils import make_logmel_spectrograms
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
#from araus1 import ARAUS_Sequence_from_npy2
from keras.models import load_model

from pydub.utils import make_chunks
from pydub import AudioSegment
#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
#model = pickle.load(open('baselinemodel.h5', 'rb'))
model = load_model('model/blmodel.h5')


#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.


@app.route('/')
def home():
    return render_template('indexn.html')

classes = 'w3-table w3-striped w3-border'

def gen_dict(df, title):
    return {'title': title,
            'table': df.head().to_html(classes=classes)
            }



@app.route('/predict', methods=['POST'])
def predict():
    x = request.form['myfile']
    filepath = os.path.abspath(x)
    #f = sf.SoundFile(filepath)
    #print("THIS IS FP", filepath)
    #filepath = filepath.replace("\\", "/")
    #print("THIS IS FP", filepath)
    global folder
    
       
    flist = filepath.replace("\\", " ").split()
                            
    print(flist)
    filename1 = flist[-1]    
    folder = flist[-2]

    print(filename1)
    print(folder)
    audio = AudioSegment.from_file(os.path.join('..', 'soundscapes', filename1))
    audio.duration_seconds == (len(audio) / 1000.0)
    
    ## set the input file with the right param and write to newnewsong.wav
    
    if audio.duration_seconds >=30:
        
        newAudio = AudioSegment.from_wav(os.path.join('..', 'soundscapes',filename1))
        newAudio = newAudio[0:30000]
        newAudio.export(os.path.join('..', 'soundscapes','ns.wav'), format="wav")
        
      #  filename = 'nns.wav' #export 30s to nns.wav
    if audio.duration_seconds <30:
        data, samplerate = sf.read(os.path.join('..', 'soundscapes', filename1))
        channels = len(data.shape)
        length_s = len(data)/float(samplerate)
        if(length_s < 30.0):
            n = math.ceil(30*samplerate/len(data))
        if(channels == 2):
            data = np.tile(data,(n,1))
        else:
            data = np.tile(data,n)
            sf.write(os.path.join('..', 'soundscapes','ns.wav'), data, samplerate)
            newAudio = AudioSegment.from_wav(os.path.join('..', 'soundscapes',"nns.wav"))
            newAudio = newAudio[0:30000]
            newAudio.export(os.path.join('..', 'soundscapes','ns.wav'), format="wav")
          
          
            
    #adjust parameters
    input_file = (os.path.join('..', 'soundscapes','ns.wav'))
    audio = AudioSegment.from_file(input_file)

    # Resample the audio to 44.1 kHz and set the number of channels to 2
    audio = audio.set_frame_rate(44100).set_channels(2)

   
# Export the modified audio to a new file
    output_file = (os.path.join('..', 'soundscapes','nns.wav'))
    audio.export(output_file, format="wav")
          
    filename = 'nns.wav'
    
    dfs=pd.DataFrame(np.array([[filename,65,1]]),
                     columns=['soundscape','insitu_leq','gain_s'])
    
    
   # s1_dir = os.path.join('..','soundscapes1')

    dfs.to_csv(os.path.join('..','soundscapes1','soundscapes1.csv'), index=False)
    soundscapes1 = pd.read_csv(os.path.join('..','soundscapes1','soundscapes1.csv'))
    # inserts the name of the user input wav file along with some attributes for processing code but these data are not used
    maskersforusers = pd.read_csv(os.path.join('..','data','maskersforusers.csv'))
    dfr = maskersforusers
    dfr.insert(0, 'soundscape', str(filename))
    dfr.insert(0, 'insitu_leq', 65)
    dfr.insert(1, 'gain_s', 1)

    dfr.insert(3, 'smr', -3)
    dfr.insert(3, 'participant', '00001')
    dfr.insert(3, 'fold_r', 1)
    dfr['stimulus_index'] = dfr.index +1
    dfr.insert(3, 'pleasant', 1)
    dfr.insert(3, 'eventful', 1)

    dfr.insert(3, 'chaotic', 1)
    dfr.insert(3, 'vibrant', 1)
    dfr.insert(3, 'uneventful', 1)
    dfr.insert(3, 'calm', 1)
    dfr.insert(3, 'annoying', 1)
    dfr.insert(3, 'monotonous', 1)
    
   # r1_dir = os.path.join('..','responses1')
    dfr.to_csv(os.path.join('..','responses1','responses1.csv'), index=False)
    responses1 = pd.read_csv(os.path.join('..','responses1','responses1.csv'))
    ## clear the folder
    

 
    dir = os.path.join('..','features2')
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
   

    
    ARAUS_Sequence_from_npy2(responses1).precompute(soundscapes1,maskersforusers)
    res_seq = ARAUS_Sequence_from_npy2(responses1, seed_val = 0, shuffle = False, verbose = 0) 
    prediction = (model.predict(res_seq))*2+3
  
    attributes = ["pleasant", 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
    
    print('for soundscape ', responses1['soundscape'][0])
    dfp = pd.DataFrame(prediction, index = [responses1['masker']], columns =['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous'])
    for i in range (8):
        dfp[attributes[i]] = dfp[attributes[i]].map('{:,.5f}'.format) #to 5 dp as difference is small
        
    mlist=[]
  
    for i in range (8):

        dfpp = dfp.sort_values(
     by=str(attributes[i]),
     ascending=False)
        
        m =  (str(dfpp.index[0]))
        m = m.replace("(", "")
        m = m.replace(")", "")
        m = m.replace(",", "")
        m = m.replace("'", "")
        mlist.append(m)
    global olist
    
    glist = []
    olist=[]
    for i in range (8):
        score = dfpp[attributes[i]][0]
        glist.append(score)
        
    
        
    thelist=[attributes, mlist, glist]
    tdf = pd.DataFrame (thelist).transpose()
    tdf.columns = ['Attributes', 'Masker', 'Score']
    
    odf = dfpp.tail(1)
    for j in range (8):
        s = odf[attributes[j]][0]
        olist.append(s)
    
  
    
    # see the maskers garners which attributes
    ndf = tdf.groupby('Masker')['Attributes'].apply(' '.join).reset_index()
  
    
    df = tdf.loc[tdf['Attributes'] == 'pleasant']
    ideal_masker = df['Masker']
    ideal_masker = ideal_masker.values[0]
    dfp.loc[ideal_masker]
    
    #plot bar chart
    global gres
    global res
    global ores
    res = [eval(i) for i in glist]
    gres = [i - min(res) for i in res]
    ores = [eval(i) for i in olist]
    
    print("Masker "+ str(ideal_masker) + " used on " + str(filename1))
    prediction_text = "The Masker which evokes an overall pleasant Soundscape is "
    pt6 = ideal_masker
    pt1 = "The Maskers that obtains the highest scores of each Attributes"
    pt2 = "The Maskers which garners the highest scored Attributes"
    pt3 = ("Masker "+ str(ideal_masker) + " used on " + str(filename1))
    pt5 = "Original score preception of your input Soundscape"
    pt4 = "Plots the difference between the lowest scored attribute vs other attribute"
    return render_template('indexn.html', pt6 = pt6, pt5 = pt5, pt4 = pt4, pt3= pt3, pt2 = pt2, pt1 = pt1, prediction_text= prediction_text, table1=[tdf.to_html(classes='data', index = False)], titles1= tdf.columns.values, table2=[ndf.to_html(classes='data', index = False)], titles2= ndf.columns.values, table3=[odf.to_html(classes='data', index = False)], titles3= odf.columns.values)
   # return render_template('indexn.html',
    #                      PageTitle = "Pandas",
     #                     (table=[df.to_html(classes='data', index = False)], titles= df.columns.values), prediction_text = 'hey')
'''
    d = {'df1': gen_dict(df, ideal_masker),
          'df2': gen_dict(ndf, 'Second Dataframe'),
          'df3': gen_dict(tdf, 'third Dataframe')
      
          }
    
   

    return render_template('indexn.html',  **d)
   # return render_template('indexn.html', prediction_text='Ideal masker is {}'.format(d))
    
'''
@app.route('/plot.png')

def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    #fig.save_fig(output)
    FigureCanvas(fig).print_png(output)
    #(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

'''
def create_figure():

    df = pd.DataFrame({
   'pig': [20, 18, 489, 675, 1776],
   'horse': [4, 25, 281, 600, 1900]
   }, index=[1990, 1997, 2003, 2009, 2014])
    
    fig = df.plot.line()
    return fig
'''

def create_figure(): ## line plot
    attributes = ["pleasant", 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
    #df=pd.DataFrame({'Score':olist, 'Attributes': attributes})

    #df2=pd.DataFrame({'Score':res, 'Attributes': attributes})

# multiple line plot
    #plt.plot('Product','Time Period 2',data=df,marker='o',color='orange',linewidth=2)
    #plt.plot('Product','Time Period 3',data=df2,marker='o',color='orange',linewidth=2)
    #plt.legend(loc='upper left')
    #pig= [20, 18, 489, 675, 1776, 4 ,4,5]

    #attributes = ["pleasant", 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
    #d = {'x': pig,
     #     'y': attributes
      #    }
   # fig = px.line(d, x="x", y="y", title="Unsorted Input") 
    fig, ax = plt.subplots(figsize = (10,8))
    fig.patch.set_facecolor('#ffffff')
    
    y = res
    x = attributes
    z = ores
    print(ores)
  
    
   # ax = df.plot.line(x = 'x', y='y')

    plt.plot(x, y, color = 'red', label = "With Masker")
    ax.plot(x, z, color = "blue", label = "Without Masker")

    plt.xticks(rotation = 30, size = 15)
    plt.ylabel("Scores", size = 15)
    plt.legend(loc="upper left")
    #plt.legend (["With Masker", "Without Masker"], loc = "upper left")

    return fig


def make_augmented_soundscapes3(responses, soundscapes, maskers,
                               mode = 'file',
                               
                               
                                # In Windows, this would be '..\\soundscapes' and in MacOSX/Linus, this would be '../soundscapes'
                               masker_dir = os.path.join('..','maskers'),
                               out_dir = os.path.join('..','soundscapes_augmented'),
                               out_format = 'wav',
                               overwrite = False,
                               stop_upon_failure = False,
                               verbose = 1):
    x = request.form['myfile']
    filepath = os.path.abspath(x)
    
    flist = filepath.replace("\\", " ").split()
                            
  
    filename = flist[-1]    
    folder = flist[-2]

    soundscape_dir = os.path.join('..',str('soundscapes'))
    # CHECK VALID MODE
    #soundscape_dir = os.path.join('..',str(folder))
    if mode not in ['file','return','both']:
        if verbose > 0: print('Warning: Invalid argument entered for mode, defaulting to "file"...')
        mode = 'file'

    # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
    if (not os.path.exists(out_dir)) and len(out_dir) > 0:
        os.makedirs(out_dir)
    
    # MAKE AUGMENTED SOUNDSCAPES
    n_failures = 0 # Will count the number of failed attempts.
    augmented_soundscapes = np.zeros((0,0,0)) # Will be used to store the augmented soundscapes.
    n_tracks = len(responses)
    if verbose > 0: print(f'Making {n_tracks} augmented soundscapes in {out_dir}...')
    for idx, (_, row) in enumerate(responses.iterrows()): # Dataframe's index may be out of order so we don't use it (and assign it to _ instead).
        if verbose > 0: print(f'Progress: {idx+1}/{n_tracks}.')
        
        try:
            # GET NECESSARY DATA FROM RESPONSES FOR CALIBRATION
            participant_id = row['participant']
            fold = row['fold_r']
            soundscape_fname = row['soundscape']
            masker_fname = row['masker']
            smr = row['smr']
            stimulus_index = row['stimulus_index']
            
            # CHECK IF FILE EXISTS
            out_fname = f'fold_{fold}_participant_{participant_id:>05}_stimulus_{stimulus_index:02d}.{out_format}'
            out_fpath = os.path.join(out_dir, out_fname)
            
            if (mode == 'file') and os.path.exists(out_fpath) and (not overwrite):
                if verbose > 0: print(f'Warning: {out_fpath} already exists, skipping its generation...') 
                continue # Skip all read and write steps to save processing time.

            # GET SOUNDSCAPE CALIBIRATION PARAMETERS
            gain_s = soundscapes[soundscapes['soundscape'] == soundscape_fname]['gain_s'].squeeze() # Gain to apply to soundscape
            leq_s = soundscapes[soundscapes['soundscape'] == soundscape_fname]['insitu_leq'].squeeze()

            # GET MASKER CALIBRATION PARAMETERS
            leq_m = leq_s - smr
            leq_m_round = np.round(leq_m,0).astype(int) # Closest integer dB value to the desired masker level that we know the correct calibration for.
            gain_c = maskers[maskers['masker'] == masker_fname][f'gain_{leq_m_round}dB'].squeeze() # This is the gain for that value,...
            leq_c = maskers[maskers['masker'] == masker_fname][f'leq_at_gain_{leq_m_round}dB'].squeeze() # and this is the Leq that was actually measured after calibration.
            gain_m = gain_c*(10**((leq_m-leq_c)/20)) # We estimate the true gain to apply to the masker within the range of 1 dB.

            # PRINT PARAMETER SUMMARY
            if verbose > 1:
                print(f'Now generating participant {participant_id}, stimulus {stimulus_index} (fold {fold}):')
                print(f'\t{soundscape_fname} (soundscape) + {masker_fname} (masker) @ SMR {smr} dB.')
                print(f'\tSoundscape Leq {leq_s:.2f} dB achieved by setting gain to {gain_s:.2e}.')
                print(f'\tMasker Leq ({leq_m:.2f} dB) achieved by setting gain to {gain_m:.2e} (interpolated from known gain {gain_c:.2e} giving Leq of {leq_c:.2f} dB).')

            # LOAD SOUNDSCAPE
            soundscape_fpath = os.path.join(soundscape_dir, soundscape_fname)
            x_s, sr_s = sf.read(soundscape_fpath)
            if not (x_s.shape == (1323000,2) and sr_s == 44100):
                if verbose > 0: print(f'Warning: Expected (1323000,2) and 44100 for soundscape shape and sampling rate but got {x_s.shape} and {sr_s}')

            # LOAD MASKER
            masker_fpath = os.path.join(masker_dir, masker_fname)
            x_m, sr_m = sf.read(masker_fpath)
            x_m = np.tile(x_m,(2,1)).T # Duplicate masker into two channels
            if not (x_m.shape == (1323000,2) and sr_s == 44100):
                if verbose > 0: print(f'Warning: Expected (1323000,2) and 44100 for masker shape (after duplication) and sampling rate but got {x_m.shape} and {sr_m}')

            # MAKE OUTPUT TRACK (= CURRENT STIMULUS)
            x = gain_s*x_s + gain_m*x_m

            # STORE SAMPLES TO OUTPUT ARRAY IF DESIRED
            if mode in ['return','both']:
                if augmented_soundscapes.shape == (0,0,0): # Means this is the first instance where generation of x was successful (possibly idx == 0 but not necessarily).
                    output_shape = (n_tracks, x.shape[0], x.shape[1])
                    if verbose > 0: print(f'Preallocating a {output_shape} array for return...')
                    augmented_soundscapes = np.zeros(output_shape) # Then we preallocate the output array (without assuming the shape of x beforehand)
                augmented_soundscapes[idx,:,:] = x
                    
            # OUTPUT TRACK TO FILE IF DESIRED
            if mode in ['file','both']:
                if os.path.exists(out_fpath):
                    if overwrite:
                        if verbose > 0: print(f'Warning: {out_fpath} already exists, overwriting it...')
                    else:
                        if verbose > 0: print(f'Warning: {out_fpath} already exists, not overwriting it...')
                        continue # Skip the writing
                if verbose > 1: print(f'\tOutputting stimulus to {out_fpath} @ {min(sr_m,sr_s)} Hz...')
                sf.write(out_fpath,x,min(sr_m,sr_s))
        except Exception as e:
            if stop_upon_failure:
                if verbose > 0: print(f'Error: Failed to make augmented soundscape #{idx+1}/{n_tracks}. Reason for failure: {e}.')
                raise
            else:
                if verbose > 0: print(f'Warning: Failed to make augmented soundscape #{idx+1}/{n_tracks}. Reason for failure: {e}. Moving on...')
                n_failures += 1
                continue
                
    return n_failures, augmented_soundscapes

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
            
   
'''

app.route('/plot.png')
def plot_png(df):
    fig = df.plot(x='', y='score', kind='bar')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

#return 'You have chosen this soundscape: %s  <br/> <a href="/">Back Home</a>' % (filepath)
    

def predict():
    
    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))

'''

#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()