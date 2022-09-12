from flask import *
# from audio_instrument_analysis import *
from warnings import filterwarnings 
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField,validators
from wtforms.validators import DataRequired

filterwarnings('ignore',category=UserWarning)

app = Flask(__name__)
app.config['SECRET_KEY']='MI5xvJ?g]K`Ezhg'
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    ty = request.form
    if ty == 'YouTube':
        return redirect( url_for('yt'))
    return render_template('index.html',ty=ty)
@app.route('/choice',methods=['GET','POST'])
def choice():
    ty = request.form.get('Type')
    if ty == 'YouTube':
        return redirect( url_for('yt'))
    ty = request.form.get('Type')
    if ty == 'File':
        return redirect( url_for('file'))

@app.route('/youtube',methods=['GET','POST'])
def yt():
    return render_template('youtube.html')

@app.route('/filename',methods=['GET','POST'])
def file():
    import os
    path = "C:/Users/jasmi/Downloads/PythonProgram/songs"
    files = os.listdir(path)
    return render_template('filename.html',files=files)

@app.route('/ytanalysis',methods=['GET','POST'])
def ytanalysis():
    ytlink = request.form['link']
    title = request.form['title']
    start_at = float(request.form['time'])
    from yt_dlp import YoutubeDL
    import os
    import ffmpeg
    ydl_opts = {
    'format':'m4a/bestaudio/best',
    'outtmpl':'songs/%(title)s.%(ext)s',
    'postprocessors':[{
        'key':'FFmpegExtractAudio',
        'preferredcodec':'wav'
        }]
    }
    with YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([ytlink])
    global source
    source = 'songs/'+title+'.wav'  
    from operator import concat
    from telnetlib import X3PAD

    def LoadData():
        from pandas import read_csv, DataFrame
        
        d = read_csv('music.csv')        ## Normalized
        dr = read_csv('rawmusic.csv')    ## Not Normalized 
        
        X = d[['cent','roll','flat','contrast','flux','bw','yin']]
        Xr = dr[['cent','roll','flat','contrast','flux','bw','yin']]
        return X, Xr, d

    def stat():
        X, Xr, d = LoadData()
        averages = Xr.mean(axis=0)
        stdevs = Xr.std(axis=0)
        return averages,stdevs

        ####    Averages for normalization
    def normalization():
        averages, stdevs = stat()
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
        return cent_av, roll_av, flat_av, contrast_av, flux_av, bw_av, yin_av, cent_sd, roll_sd, flat_sd, contrast_sd, flux_sd, bw_sd, yin_sd

    def SongFeatures(src=source):
        from librosa import feature, load, onset, yin,note_to_hz
        from numpy import asarray
        from pandas import DataFrame, concat
        
        lowNote = note_to_hz('C2')
        highNote = note_to_hz('C8')

        Y, sr = load(src, mono=True, duration=5.0, offset=start_at)
        cent_av, roll_av, flat_av, contrast_av, flux_av, bw_av, yin_av, cent_sd, roll_sd, flat_sd, contrast_sd, flux_sd, bw_sd, yin_sd = normalization()
        Y, sr = Y, sr

        centroid = asarray(feature.spectral_centroid(y=Y, sr=sr))[0]
        rolloff = asarray(feature.spectral_rolloff(y=Y, sr=sr))[0]
        flatness = asarray(feature.spectral_flatness(y=Y))[0]
        contrast = asarray(feature.spectral_contrast(y=Y, sr=sr))[0]
        flux = asarray(onset.onset_strength(y=Y, sr=sr))
        bandwidth = asarray(feature.spectral_bandwidth(y=Y, sr=sr))[0]
        yins = asarray(yin(y=Y, fmin=lowNote, fmax=highNote, sr=sr))

        ####    Normalize song features
        centNorm = [((i-cent_av)/cent_sd) for i in centroid]
        centNorm = DataFrame(data=centNorm)

        rollNorm = [((i-roll_av)/roll_sd) for i in rolloff]
        rollNorm = DataFrame(data=rollNorm)

        flatNorm = [((i-flat_av)/flat_sd) for i in flatness]
        flatNorm = DataFrame(data=flatNorm)

        contNorm = [((i-contrast_av)/contrast_sd) for i in contrast]
        contNorm = DataFrame(data=contNorm)

        fluxNorm = [((i-flux_av)/flux_sd) for i in flux]
        fluxNorm = DataFrame(data=fluxNorm)

        bwNorm = [((i-bw_av)/bw_sd) for i in bandwidth]
        bwNorm = DataFrame(data=bwNorm)

        yinNorm = [((i-yin_av)/yin_sd) for i in yins]
        yinNorm = DataFrame(data=yinNorm)

        global features
        features = concat([centNorm,rollNorm,flatNorm,contNorm,fluxNorm,bwNorm,yinNorm],axis=1,ignore_index=True)
    SongFeatures()

    class ia:
        def __init__(self, instrument):
            self.instrument=instrument
        def inst(self):
            from statistics import mode
            from sklearn.ensemble import ExtraTreesClassifier
            et = ExtraTreesClassifier(random_state=0,n_jobs=-1,max_depth=50)
            X, Xr, d = LoadData()
            
            y = d[self.instrument]
            y_pred = et.fit(X,y).predict(features)
            if mode(y_pred)==1:
                return True
            else:
                return False

    def list():
        instruments = []
        piano,unpitched,organ,string,voice,brass,reed,wind,pitched,guitar,synth = ia('piano'),ia('unpitched'),ia('organ'),ia('string'),ia('voice'),ia('brass'),ia('reed'),ia('wind'),ia('pitched'),ia('guitar'),ia('synth')

        if piano.inst()==True:
            instruments.append('piano')
        if unpitched.inst()==True:
            instruments.append('unpitched percussion')
        if organ.inst()==True:
            instruments.append('organ')
        if string.inst()==True:
            instruments.append('strings')
        if voice.inst()==True:
            instruments.append('voice')
        if brass.inst()==True:
            instruments.append('brass')
        if reed.inst()==True:
            instruments.append('reeds')
        if wind.inst()==True:
            instruments.append('winds')
        if pitched.inst()==True:
            instruments.append('pitched percussion')
        if guitar.inst()==True:
            instruments.append('guitar')
        if synth.inst()==True:
            instruments.append('synth')
        return instruments

    instrument_list = list()
    return render_template('ytanalysis.html',instrument_list=instrument_list,title=title)

@app.route('/fileanalysis',methods=['GET','POST'])
def fileanalysis():
    title = request.form['title']
    global source
    start_at = float(request.form['time'])

    source = 'songs/'+title+'.wav'  
    def LoadData():
        from pandas import read_csv, DataFrame
        
        d = read_csv('music.csv')        ## Normalized
        dr = read_csv('rawmusic.csv')    ## Not Normalized 
        
        X = d[['cent','roll','flat','contrast','flux','bw','yin']]
        Xr = dr[['cent','roll','flat','contrast','flux','bw','yin']]
        return X, Xr, d

    def stat():
        X, Xr, d = LoadData()
        averages = Xr.mean(axis=0)
        stdevs = Xr.std(axis=0)
        return averages,stdevs

        ####    Averages for normalization
    def normalization():
        averages, stdevs = stat()
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
        return cent_av, roll_av, flat_av, contrast_av, flux_av, bw_av, yin_av, cent_sd, roll_sd, flat_sd, contrast_sd, flux_sd, bw_sd, yin_sd

    def SongFeatures(src=source):
        from librosa import feature, load, onset, yin,note_to_hz
        from numpy import asarray
        from pandas import DataFrame, concat
        
        lowNote = note_to_hz('C2')
        highNote = note_to_hz('C8')

        Y, sr = load(src, mono=True, duration=5.0, offset=start_at)
        cent_av, roll_av, flat_av, contrast_av, flux_av, bw_av, yin_av, cent_sd, roll_sd, flat_sd, contrast_sd, flux_sd, bw_sd, yin_sd = normalization()
        Y, sr = Y, sr

        centroid = asarray(feature.spectral_centroid(y=Y, sr=sr))[0]
        rolloff = asarray(feature.spectral_rolloff(y=Y, sr=sr))[0]
        flatness = asarray(feature.spectral_flatness(y=Y))[0]
        contrast = asarray(feature.spectral_contrast(y=Y, sr=sr))[0]
        flux = asarray(onset.onset_strength(y=Y, sr=sr))
        bandwidth = asarray(feature.spectral_bandwidth(y=Y, sr=sr))[0]
        yins = asarray(yin(y=Y, fmin=lowNote, fmax=highNote, sr=sr))

        ####    Normalize song features
        centNorm = [((i-cent_av)/cent_sd) for i in centroid]
        centNorm = DataFrame(data=centNorm)

        rollNorm = [((i-roll_av)/roll_sd) for i in rolloff]
        rollNorm = DataFrame(data=rollNorm)

        flatNorm = [((i-flat_av)/flat_sd) for i in flatness]
        flatNorm = DataFrame(data=flatNorm)

        contNorm = [((i-contrast_av)/contrast_sd) for i in contrast]
        contNorm = DataFrame(data=contNorm)

        fluxNorm = [((i-flux_av)/flux_sd) for i in flux]
        fluxNorm = DataFrame(data=fluxNorm)

        bwNorm = [((i-bw_av)/bw_sd) for i in bandwidth]
        bwNorm = DataFrame(data=bwNorm)

        yinNorm = [((i-yin_av)/yin_sd) for i in yins]
        yinNorm = DataFrame(data=yinNorm)

        global features
        features = concat([centNorm,rollNorm,flatNorm,contNorm,fluxNorm,bwNorm,yinNorm],axis=1,ignore_index=True)
    SongFeatures()

    class ia:
        def __init__(self, instrument):
            self.instrument=instrument
        def inst(self):
            from statistics import mode
            from sklearn.ensemble import ExtraTreesClassifier
            et = ExtraTreesClassifier(random_state=0,n_jobs=-1,max_depth=50)
            X, Xr, d = LoadData()
            
            y = d[self.instrument]
            y_pred = et.fit(X,y).predict(features)
            if mode(y_pred)==1:
                return True
            else:
                return False

    def list():
        instruments = []
        piano,unpitched,organ,string,voice,brass,reed,wind,pitched,guitar,synth = ia('piano'),ia('unpitched'),ia('organ'),ia('string'),ia('voice'),ia('brass'),ia('reed'),ia('wind'),ia('pitched'),ia('guitar'),ia('synth')

        if piano.inst()==True:
            instruments.append('piano')
        if unpitched.inst()==True:
            instruments.append('unpitched percussion')
        if organ.inst()==True:
            instruments.append('organ')
        if string.inst()==True:
            instruments.append('strings')
        if voice.inst()==True:
            instruments.append('voice')
        if brass.inst()==True:
            instruments.append('brass')
        if reed.inst()==True:
            instruments.append('reeds')
        if wind.inst()==True:
            instruments.append('winds')
        if pitched.inst()==True:
            instruments.append('pitched percussion')
        if guitar.inst()==True:
            instruments.append('guitar')
        if synth.inst()==True:
            instruments.append('synth')
        return instruments

    instrument_list = list()
    return render_template('ytanalysis.html',instrument_list=instrument_list,start_at=start_at,title=title)





