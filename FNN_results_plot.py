import random , os , torch , json
from Brain.brain import NeuralNet
from Brain.neural_network import bag_of_words , tokenize
from colorama import Fore
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#===========================================================================================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

FILE = "D:\\A.I\\FNN\\1\\Train\\TrainData.pth"
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
response_times = []
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# vosk engine for speech recognition recognition ===========================================================================================================================================================================================================================================================================================
iteration = 0  # Initialize iteration counter

while iteration < 5:  # Run for 5 iterations
    start_time = time.time()  # Record the start time
    
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
    actual_tag = "introduction" # Replace with the actual tag
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
    
    end_time = time.time()  # Record the end time
    response_time = end_time - start_time
    response_times.append(response_time)  # Store response time

    iteration += 1  # Increment iteration counter

# Print the recorded response times
for i, time in enumerate(response_times, 1):
    print(f"Iteration {i}: Response Time = {time} seconds")


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


# Plotting the response time
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), response_times, marker='o', color='b')
plt.title('Response Time Plot')
plt.xlabel('Interaction Number')
plt.ylabel('Response Time (seconds)')
plt.grid(True)
plt.show()


# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plotting accuracy, precision, recall, and F1-score
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
sns.barplot(x=metrics, y=values, palette='viridis')
#sns.barplot(x='Metrics', y='Values', hue='Metrics', data=pd.DataFrame({'Metrics': metrics, 'Values': values}), palette='viridis', legend=False)
plt.title('Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.show()