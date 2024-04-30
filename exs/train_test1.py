import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Brain.neural_network import bag_of_words, tokenize, stem
from Brain.brain import NeuralNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Process intents data to create training dataset
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = [',', '?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)

# Define training parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

# Define custom dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):    
        return self.n_samples
    
# Create dataset and data loader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Define device for model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, criterion, and optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for words, labels in train_loader:        
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) * 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final Loss: {loss.item():.4f}')

# Save trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "TrainData.pth"
torch.save(data, FILE)
print(f"Training Complete, Model Saved To {FILE}")

# Load trained model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load(FILE))
model.eval()

# Initialize lists to store predictions and actual labels
predicted_tags = []
actual_tags = []

# Iterate through test data (similar to the training loop)
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X = torch.from_numpy(np.array(bag)).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=0)
    predicted_tag = tags[predicted.item()]

    # Store predicted and actual tags
    predicted_tags.append(predicted_tag)
    actual_tags.append(tag)

# Compute evaluation metrics
accuracy = accuracy_score(actual_tags, predicted_tags)
precision = precision_score(actual_tags, predicted_tags, average='weighted', zero_division=1)
recall = recall_score(actual_tags, predicted_tags, average='weighted', zero_division=1)
f1 = f1_score(actual_tags, predicted_tags, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(actual_tags, predicted_tags)

# Print evaluation metrics
print("****************************")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("****************************")
print("Confusion Matrix:\n", conf_matrix)
