import random , os , torch , json , pyaudio , openai
from Brain.brain import NeuralNet
from Brain.neural_network import bag_of_words , tokenize
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

# vosk engine for speech recognition recognition ===========================================================================================================================================================================================================================================================================================
while True:
    sentence = input(Fore.RED + "User said:- ")
    
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
                
                print(reply)