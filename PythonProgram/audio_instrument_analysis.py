from statistics import mode
from telnetlib import X3PAD
import librosa as _lr
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import warnings

inp = input('Type file name (without the extension): ')
file = '.'.join([inp,'wav'])

slash = R"\\"
src = slash.join(['instruments',file])
    ####    stfu
warnings.filterwarnings('ignore',category=UserWarning)

    ####    Load the data
d = pd.read_csv('music.csv')        ## Normalized
dr = pd.read_csv('rawmusic.csv')    ## Not Normalized 

    ####    Define X
X = d[['cent','roll','flat','contrast','flux','bw','yin']]
Xr = dr[['cent','roll','flat','contrast','flux','bw','yin']]

averages = Xr.mean(axis=0)
stdevs = Xr.std(axis=0)

    ####    Averages for normalization
cent_av = averages.take([0],axis=0)
roll_av = averages.take([1],axis=0)
flat_av = averages.take([2],axis=0)
contrast_av = averages.take([3],axis=0)
flux_av = averages.take([4],axis=0)
bw_av = averages.take([5],axis=0)
yin_av = averages.take([6],axis=0)

    ####    Standard Deviations for Normalization
cent_sd = stdevs.take([0],axis=0)
roll_sd = stdevs.take([1],axis=0)
flat_sd = stdevs.take([2],axis=0)
contrast_sd = stdevs.take([3],axis=0)
flux_sd = stdevs.take([4],axis=0)
bw_sd = stdevs.take([5],axis=0)
yin_sd = stdevs.take([6],axis=0)


    ####    DEFINE LOW AND HIGH NOTES
lowNote = _lr.note_to_hz('C2')
highNote = _lr.note_to_hz('C8')

    ####    Extract song features
Y, sr = _lr.load(src, mono=True, duration=5.0, offset=1)

centroid = np.asarray(_lr.feature.spectral_centroid(y=Y, sr=sr))[0]
rolloff = np.asarray(_lr.feature.spectral_rolloff(y=Y, sr=sr))[0]
flatness = np.asarray(_lr.feature.spectral_flatness(y=Y))[0]
contrast = np.asarray(_lr.feature.spectral_contrast(y=Y, sr=sr))[0]
flux = np.asarray(_lr.onset.onset_strength(y=Y, sr=sr))
bandwidth = np.asarray(_lr.feature.spectral_bandwidth(y=Y, sr=sr))[0]
yins = np.asarray(_lr.yin(y=Y, fmin=lowNote, fmax=highNote, sr=sr))

####    Normalize song features
centNorm = []
for i in centroid:
    c = (i-cent_av)/cent_sd
    centNorm.append(c)
centNorm = pd.DataFrame(data=centNorm)

rollNorm = []
for i in rolloff:
    c = (i-roll_av)/roll_sd
    rollNorm.append(c)
rollNorm = pd.DataFrame(data=rollNorm)

flatNorm = []
for i in flatness:
    c = (i-flat_av)/flat_sd
    flatNorm.append(c)
flatNorm = pd.DataFrame(data=flatNorm)

contNorm = []
for i in contrast:
    c = (i-contrast_av)/contrast_sd
    contNorm.append(c)
contNorm = pd.DataFrame(data=contNorm)

fluxNorm = []
for i in flux:
    c = (i-flux_av)/flux_sd
    fluxNorm.append(c)
fluxNorm = pd.DataFrame(data=fluxNorm)

bwNorm = []
for i in bandwidth:
    c = (i-bw_av)/bw_sd
    bwNorm.append(c)
bwNorm = pd.DataFrame(data=bwNorm)

yinNorm = []
for i in yins:
    c = (i-yin_av)/yin_sd
    yinNorm.append(c)
yinNorm = pd.DataFrame(data=yinNorm)

features = pd.concat([centNorm,rollNorm,flatNorm,contNorm,fluxNorm,bwNorm,yinNorm],axis=1,ignore_index=True)

####    Define algorithm
et = ExtraTreesClassifier(random_state=0)

####    Extra Trees
def p():
    y = d['piano']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        piano = True
    else:
        piano = False
    return piano

def u():
    y = d['unpitched']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        unpitched = True
    else:
        unpitched = False
    return unpitched

def o():
    y = d['organ']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        organ = True
    else:
        organ = False
    return organ

def s():
    y = d['string']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        string = True
    else:
        string = False
    return string

def v():
    y = d['voice']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        voice = True
    else:
        voice = False
    return voice

def b():
    y = d['brass']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        brass = True
    else:
        brass = False
    return brass

def r():
    y = d['reed']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        reed = True
    else:
        reed = False
    return reed

def w():
    y = d['wind']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        wind = True
    else:
        wind = False
    return wind

def pt():
    y = d['pitched']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        pitched = True
    else:
        pitched = False
    return pitched

def g():
    y = d['guitar']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        guitar = True
    else:
        guitar = False
    return guitar

def sn():
    y = d['synth']
    y_pred = et.fit(X,y).predict(features)
    if mode(y_pred)==1:
        synth = True
    else:
        synth = False
    return synth

instruments = []
if p()==True:
    instruments.append('Piano')
if u()==True:
    instruments.append('Unpitched')
if o()==True:
    instruments.append('Organ')
if s()==True:
    instruments.append('Strings')
if v()==True:
    instruments.append('Voice')
if b()==True:
    instruments.append('Brass')
if r()==True:
    instruments.append('Reeds')
if w()==True:
    instruments.append('Winds')
if pt()==True:
    instruments.append('Pitched')
if g()==True:
    instruments.append('Guitar')
if sn()==True:
    instruments.append('Synth')

print(instruments)

input('Press Enter to Continue...')
####    Metrics
# acc = metrics.accuracy_score(piano,y_pred)
# prec = metrics.precision_score(piano,y_pred)
# sens = metrics.recall_score(piano,y_pred)

# print("Accuracy: ",acc)
# print("Precision: ",prec)
# print("Sensitivity:",sens)





# has_piano = []
# for i in bandwidth:
#     has_piano.append(1)

# has_unpitched = []
# for i in bw:
#     has_unpitched.append(0)

# has_organ = []
# for i in bw:
#     has_organ.append(0)

# has_synth = []
# for i in bw:
#     has_synth.append(0)

# has_voice = []
# for i in bw:
#     has_voice.append(0)

# has_brass = []
# for i in bw:
#     has_brass.append(0)

# has_reed = []
# for i in bw:
#     has_reed.append(0)

# has_wind = []
# for i in bw:
#     has_wind.append(0)

# has_pitch = []
# for i in bw:
#     has_pitch.append(0)

# has_guit = []
# for i in bw:
#     has_guit.append(0)

# has_elec = []
# for i in bw:
#     has_elec.append(0)
# piano = pd.DataFrame(np.transpose(has_piano))