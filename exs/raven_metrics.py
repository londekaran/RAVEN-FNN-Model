import random , os , torch , json , pyaudio , openai
from Brain.brain import NeuralNet
from Brain.neural_network import bag_of_words , tokenize
from vosk import Model, KaldiRecognizer
from colorama import Fore
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#===========================================================================================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)
    
# Load test data
# test_data = load_test_data()  # Load your test dataset

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

# Initialize lists to store predictions and actual labels
predicted_tags = []
actual_tags = []

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# vosk engine for speech recognition recognition ===========================================================================================================================================================================================================================================================================================
iteration = 0  # Initialize iteration counter

while iteration < 5:  # Run for 5 iterations
    sentence = input(Fore.RED + "User said:- ")
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _ , predicted = torch.max(output,dim = 1)
    tag = tags[predicted.item()]
    
    # Store predicted tag
    predicted_tags.append(tag)
    
    # Get actual tag (for testing, replace this with actual tag from your test dataset)
    actual_tag = "name"  # Replace with the actual tag
    actual_tags.append(actual_tag)
    
    probs = torch.softmax(output,dim = 1)
    prob = probs[0][predicted.item()]
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #Predicted Execution =============================================================================================================================================================================================
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                
                print(reply)
    
    iteration += 1  # Increment iteration counter

# Calculate evaluation metrics
accuracy = accuracy_score(actual_tags, predicted_tags)
precision = precision_score(actual_tags, predicted_tags, average='weighted', zero_division=1)
recall = recall_score(actual_tags, predicted_tags, average='weighted', zero_division=1)
f1 = f1_score(actual_tags, predicted_tags, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(actual_tags, predicted_tags)

print("****************************")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("****************************")
print("Confusion Matrix:\n", conf_matrix)