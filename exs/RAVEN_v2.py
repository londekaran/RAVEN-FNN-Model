#--------------------------------------------------------------------------------------------------------#
#   |------\   |------|   \        /  |------  |\      |                                                 #
#   |      /   |      |    \      /   |        | \    |                                                  #
#   |    /     |------|     \    /    |-----   |  \   |                                                  #
#   |   \      |      |      \  /     |        |   \ |                                                   #
#   |    \     |      |       V       |-----   |    \|               V3.0  --CREATED BY:- KARAN.N.LONDE  #
#                                                                                                        #
# RESPONSIVE     AI        VOICE    ENABLED   NETWORK                                                    #
#--------------------------------------------------------------------------------------------------------#

import random , os , torch , json , pyaudio , openai
from Brain.brain import NeuralNet
from Brain.neural_network import bag_of_words , tokenize
from vosk import Model, KaldiRecognizer
from colorama import Fore
import time

#===========================================================================================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

FILE = "D:\\A.I\\CNN\\1\\Train\\TrainData.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Name = "raven"

# VOSK model and details ======================================================================================================================================================================================
model_path = r"D:\\A.I\\F R I D A Y\\FRIDAY_v4\\models\\vosk-model-en-in-0.5"
vosk_model = Model(model_path)
sample_rate = 16000
data_format = pyaudio.paInt16
stream = pyaudio.PyAudio().open(format=data_format, channels=1, rate=sample_rate, input=True, frames_per_buffer=8192)
stream.start_stream()
recognizer = KaldiRecognizer(vosk_model,sample_rate)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

total_queries = 0
correct_predictions = 0
total_response_time = 0

# vosk engine for speech recognition recognition ===========================================================================================================================================================================================================================================================================================
while True:
    data=stream.read(2000, exception_on_overflow = False)
    if len(data)==0:
        break
    if recognizer.AcceptWaveform(data):
        start_time = time.time()
        result = json.loads(recognizer.Result())
        query = result.get("text" , "")
        print(Fore.RED + "User said:- " , query)
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #Prediction of tokens =================================================================================================================================================================================
        sentence = query
        result = str(sentence)
        
        if sentence == 'bye':
            exit()
        
        sentence = tokenize(sentence)
        X = bag_of_words(sentence,all_words)
        X = X.reshape(1,X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model(X)
        _ , predicted = torch.max(output,dim = 1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output,dim = 1)
        prob = probs[0][predicted.item()]
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #Predicted Execution =============================================================================================================================================================================================
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    reply = random.choice(intent["responses"])
                    #non input query----------------------------------------------------------------------------------------------------------------------
                
                    print(reply)