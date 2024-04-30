from vosk import Model, KaldiRecognizer
import json
import pyaudio

# VOSK model and details======================================================================================================================================================================================
model_path = r"D:\\A.I\\F R I D A Y\\FRIDAY_v4\\models\\vosk-model-en-in-0.5"
model = Model(model_path)
sample_rate = 16000
data_format = pyaudio.paInt16
stream = pyaudio.PyAudio().open(format=data_format, channels=1, rate=sample_rate, input=True, frames_per_buffer=8192)
stream.start_stream()
recognizer = KaldiRecognizer(model,sample_rate)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# vosk engine for speech recognition recognition===========================================================================================================================================================================================================================================================================================
def Listen():
    
    while True:
        data=stream.read(2000, exception_on_overflow = False)
        if len(data)==0:
            break
        if recognizer.AcceptWaveform(data):
            #result = json.loads(recognizer.Result())
            #query = result.get("text" , "")
            #print("User said:- " , query)
            
            query=recognizer.Result()
            #query=json.loads(query)
            print('User said: ' + query)
            
    query = str(query)
    return query.lower()

#---------------------------------------------------------------------------------------------------------------------------
