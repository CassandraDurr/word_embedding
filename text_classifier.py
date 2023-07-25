"""A module performing sentence classification using trained word embeddings from the word_embeddings_torch module."""
import json

import numpy as np
import torch
import torch.nn as nn

from utils import (
    SentenceMLP,
    convert_data,
    data_preprocessing,
    labels_numerical,
    text_2_num,
    to_TensorDataset,
    test_confusion_matrix,
)

# Create an output file to store info in
filename = "output_we_text_classifier"
config = "not_normalised"
with open(f"out/{filename}.txt", "w") as output_file:
    output_file.write("Text classification output:\n\n")

# Call in trained embeddings
with open("out/embeddings_5_5_False.npy", "rb") as f:
    embeddings = np.load(f)

vocab_size, embedding_dim = embeddings.shape
print(f"Shape of embedding matrix = {embeddings.shape}")

# Call data in
with open("data/train.json", "r") as json_file:
    train_data = json.load(json_file)
with open("data/test.json", "r") as json_file:
    test_data = json.load(json_file)
with open("data/validation.json", "r") as json_file:
    validation_data = json.load(json_file)

# Recreate the training dictionary (word:num)
with open("data/train_texts.txt", "r") as file:
    train_txt = file.readlines()
train = data_preprocessing(train_txt)
train_dictionary = text_2_num(train)

# Convert label data to one-hot encoding
train_data["numerical_labels"] = labels_numerical(labels=train_data["labels"])
validation_data["numerical_labels"] = labels_numerical(labels=validation_data["labels"])
test_data["numerical_labels"] = labels_numerical(labels=test_data["labels"])

# Convert numerical data to sentence embeddings
train_data["sentence_embeddings"] = np.array(
    convert_data(data=train_data["data"], embedding=embeddings)
)
validation_data["sentence_embeddings"] = np.array(
    convert_data(data=validation_data["data"], embedding=embeddings)
)
test_data["sentence_embeddings"] = np.array(
    convert_data(data=test_data["data"], embedding=embeddings)
)

# Parameters
batch_size = 64
num_epochs = 100

# Early stopping parameters
patience = 4
best_validation_loss = 199804042023
num_epochs_no_improvement = 0

# Prepare data
train_dataset = to_TensorDataset(train_data)
validation_dataset = to_TensorDataset(validation_data)
test_dataset = to_TensorDataset(test_data)
training_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, shuffle=True, batch_size=batch_size
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=batch_size
)

# Build a simple multi-layer perceptron for text classification with categorical cross-entropy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence_classifier = SentenceMLP(
    input_size=embedding_dim, nodes_per_layer=[256, 128, 64], num_classes=4
).to(device)
optimiser = torch.optim.Adam(sentence_classifier.parameters(), lr=0.001)
cross_entropy = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    sentence_classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        output = sentence_classifier(inputs)

        # Compute the loss
        loss = cross_entropy(output, labels)
        loss.backward()

        # Optimiser step
        optimiser.step()
        optimiser.zero_grad()

        # Update running loss and accuracy
        running_loss += loss.item()

        # Increment number of correct classifications
        _, predicted = torch.max(output.data, 1)
        _, truth = torch.max(labels, 1)
        correct += (predicted == truth).sum().item()

        # Increment total with the number of elements in the batch
        total += labels.size(0)

    # Validation
    with torch.no_grad():
        running_vloss = 0.0
        v_total = 0
        v_correct = 0
        for vi, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = sentence_classifier(vinputs)
            vloss = cross_entropy(voutputs, vlabels)
            running_vloss += vloss
            # Increment total with the number of elements in the batch
            v_total += vlabels.size(0)
            # Increment number of correct classifications
            _, v_predicted = torch.max(voutputs.data, 1)
            _, v_truth = torch.max(vlabels, 1)
            v_correct += (v_predicted == v_truth).sum().item()

    train_accuracy = 100 * correct / total
    valid_accuracy = 100 * v_correct / v_total
    print(
        f"Epoch {epoch + 1}/{num_epochs}: Training loss: {running_loss/(i+1)}, Validation loss: {running_vloss.item()/(vi+1)}"
    )
    print(
        f"Epoch {epoch + 1}/{num_epochs}: Training accuracy: {train_accuracy}%, Validation accuracy: {valid_accuracy}%"
    )
    with open(f"out/{filename}.txt", "a") as output_file:
        output_file.write(
            f"\nEpoch {epoch + 1}/{num_epochs}: Training loss: {running_loss/(i+1)}, Validation loss: {running_vloss.item()/(vi+1)}"
        )
        output_file.write(
            f"\nEpoch {epoch + 1}/{num_epochs}: Training accuracy: {train_accuracy}%, Validation accuracy: {valid_accuracy}%"
        )

    # Check for early stopping
    if running_vloss.item() / (vi + 1) < best_validation_loss:
        best_validation_loss = running_vloss.item() / (vi + 1)
        num_epochs_no_improvement = 0
    else:
        num_epochs_no_improvement += 1

    if num_epochs_no_improvement >= patience:
        print("Early stopping triggered.")
        with open(f"out/{filename}.txt", "a") as output_file:
            output_file.write(f"\nEarly stopping triggered.")
        break

# Test evaluation

sentence_classifier.eval()
correct = 0
total = 0

# Initialise an empty confusion matrix
class_labels = ["Sports", "World", "Sci/Tech", "Business"]
conf_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = sentence_classifier(inputs)
        _, predicted = torch.max(outputs.data, 1)
        _, truth = torch.max(labels, 1)

        # Increment total with the number of elements in the batch
        total += labels.size(0)
        # Increment number of correct classifications
        correct += (predicted == truth).sum().item()

        # Update the confusion matrix
        for t, p in zip(truth, predicted):
            conf_matrix[t][p] += 1

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy}%")
with open(f"out/{filename}.txt", "a") as output_file:
    output_file.write(f"Test Accuracy: {accuracy}%")

# Plot the confusion matrix as a heatmap
test_confusion_matrix(
    conf_matrix=conf_matrix,
    class_labels=class_labels,
    save_location="out",
    save_name=config,
)
