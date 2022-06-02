import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import librosa
from tensorflow.keras.models import load_model 
from pickle import load
import wave
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import base64
app = Flask(__name__)


RESPEAKER_RATE = 16000 #16KHz data rate in android devices
RESPEAKER_CHANNELS = 4 #4 input in a standart android device
RESPEAKER_WIDTH = 2 
RESPEAKER_INDEX = 0  # refer to input device id
CHUNK = 1024 #recording frequency
RECORD_SECONDS = 3 
WAVE_OUTPUT_FILENAME = "output.wav"


model = load_model('classifier.h5', compile = False) #calling pre-trained model ready
model.load_weights('classifier_weights') #calling the weigths ready
tran = load(open('tran.pkl', 'rb')) #tensorflow model saved in a pickle file
classes = ["air_conditioner", "car_horn", "children_playing", "dsog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"] #feature labels are set in a dictionary


#to display the connection status
@app.route('/', methods=['GET'])
def handle_call():
    return "Successfully Connected"


@app.route('/getdata', methods=['POST'])
def getdata():
    json_data = request.json
    val = json_data["data"]


    #a = write(WAVE_OUTPUT_FILENAME, RESPEAKER_RATE, val)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb') #opens the filestream obtained from GET request and converted into .wav
    wf.setsampwidth(2) #because the trained dataset objects have those values
    wf.setframerate(RESPEAKER_RATE)#because the trained dataset objects have those values
    wf.writeframes(b''.join(val)) # audio frame value (basically sample rate)
    wf.close

        
        
    f_name='output.wav'
    X, s_rate = librosa.load(f_name, res_type='kaiser_fast') #in order to use librosa soundfile features, we load it
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T,axis=0) #Compute the arithmetic mean along the specified axis with Mel-frequency cepstral coefficients (MFCCs)
    try:
            t = np.mean(librosa.feature.tonnetz(
                           y=librosa.effects.harmonic(X),
                           sr=s_rate).T,axis=0)  #Computes the tonal centroid features (tonnetz)
    except:
        print(f_name)  
    m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T,axis=0) #mean obtained by a mel-scaled spectrogram.
    s = np.abs(librosa.stft(X)) #absolute value Short-time Fourier transform (STFT) elementwise.
    c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T,axis=0) #mean obtained by a chromagram from a waveform or power spectrogram.
        
    feature = [np.concatenate((m, mf, t, c), axis=0) ] #Join a sequence of arrays along an existing axis.
    feat = tran.transform(feature)  #pickle library transform our features into human readable format and 1 row format
        
    prediction = model.predict(feat)[0] #use the pre-trained model to predict input which is transformed into a predictable format.
    print("This is predicted to be ", classes[np.where(prediction == max(prediction))[0][0]], " with the possibility of ", max(prediction)) #it prints out the most accurate prediction between labels.
        
    msg = "0:none" #if the accuracy match rate is less than %90, we dont label the sound, it is indefinite
    if (max(prediction)>0.9):
        msg = "1:" + classes[np.where(prediction == max(prediction))[0][0]]			 #if there is a match, we send a label
            
    
    return msg

if __name__ == '__main__':
   app.run()
