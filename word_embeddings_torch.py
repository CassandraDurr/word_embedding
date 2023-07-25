"""A module which trains a skip-gram word embedding model with negative sampling using the PyTorch framework."""
import json
import re
import time

import numpy as np
import torch
import torch.nn as nn

from utils import (
    SkipGramNegativeSampling,
    context_window,
    convert_text_2_number,
    create_dataset,
    data_preprocessing,
    get_val_train_datasets,
    text_2_num,
)

# ------------------------------------------
# Output storage
# ------------------------------------------

filename = "output_we_torch"
with open(f"out/{filename}.txt", "w") as output_file:
    output_file.write("Word embedding model output:\n\n")

# ------------------------------------------
# Data ingestion and processing
# ------------------------------------------
start_time = time.time()

# Ingest and process data
# Train data
with open("data/train_texts.txt", "r") as file:
    train_txt = file.readlines()
train = data_preprocessing(train_txt)
# Train labels
with open("data/train_labels.txt", "r") as file:
    train_labels = file.readlines()
# Remove newlines
train_labels = [re.sub("\n", "", string) for string in train_labels]
# Test data
with open("data/test_texts.txt", "r") as file:
    test = file.readlines()
test = data_preprocessing(test)
# Test labels
with open("data/test_labels.txt", "r") as file:
    test_labels = file.readlines()
# Remove newlines
test_labels = [re.sub("\n", "", string) for string in test_labels]

# ------------------------------------------
# Convert text data to numbers
# + Train/validation split
# ------------------------------------------
# Build dictionary from training tokens, and add <UNK> token
train_dictionary = text_2_num(train)

# If you already have the data
data_exists = True
if data_exists:
    # Load data
    with open("data/train.json", "r") as json_file:
        train_data = json.load(json_file)
    with open("data/test.json", "r") as json_file:
        test_data = json.load(json_file)
    with open("data/validation.json", "r") as json_file:
        validation_data = json.load(json_file)
else:
    train = convert_text_2_number(train, train_dictionary)
    test = convert_text_2_number(test, train_dictionary)

    # Do a train/ validation split
    np.random.seed(20230718)
    split = np.random.choice(a=[0, 1], p=[0.85, 0.15], size=len(train_labels))
    train_data = {"data": [], "labels": []}
    validation_data = {"data": [], "labels": []}
    test_data = {"data": test, "labels": test_labels}
    for cnt, itm in enumerate(split):
        if itm == 0:
            train_data["data"].append(train[cnt])
            train_data["labels"].append(train_labels[cnt])
        else:
            validation_data["data"].append(train[cnt])
            validation_data["labels"].append(train_labels[cnt])

    # Check if lengths all match
    assert len(test_data["data"]) == len(
        test_data["labels"]
    ), "Test data and labels have different lengths."
    assert len(train_data["data"]) == len(
        train_data["labels"]
    ), "Train data and labels have different lengths."
    assert len(validation_data["data"]) == len(
        validation_data["labels"]
    ), "Validation data and labels have different lengths."

    # Save data
    with open("data/train.json", "w") as json_file:
        json.dump(train_data, json_file)
    with open("data/test.json", "w") as json_file:
        json.dump(test_data, json_file)
    with open("data/validation.json", "w") as json_file:
        json.dump(validation_data, json_file)

# Write lengths to output
with open(f"out/{filename}.txt", "a") as output_file:
    output_file.write("Dataset sizes:\n")
    output_file.write(f"Train: {len(train_data['data'])}")
    output_file.write(f"\nValidation: {len(validation_data['data'])}")
    output_file.write(f"\nTest: {len(test_data['data'])}\n")

print(
    f"Data injestion and processing complete. Time {time.time() - start_time} seconds"
)

# ------------------------------------------
# Data setup for:
# Vanilla skip-gram with negative sampling
# ------------------------------------------
# Get vocabulary size
vocab_size = len(train_dictionary)
num_pos_neg = 1  # number of times drawing positives and negatives
batch_size = 4096
num_epochs = 100

# Early stopping parameters
patience = 4
best_validation_loss = 199804042023
num_epochs_no_improvement = 0

# Steps to speed up training because it is super slow
# Remove anomaly detection
torch.autograd.set_detect_anomaly(False)
# Use gradient accumulation
# https://discuss.pytorch.org/t/accumulating-gradients/30020
accumulation_steps = 64
embedding_dim = 512

# +++++++ Hyperparameters +++++++
window = 5
num_negatives = 5  # per positive pair
normalise_embeddings = False
config = f"{window}_{num_negatives}_{normalise_embeddings}"
# +++++++++++++++++++++++++++++++

start_time = time.time()

# Get paired data from context windows
train_combinations = context_window(data=train_data["data"], window=window)
valid_combinations = context_window(data=validation_data["data"], window=window)
# Get key-item pairs where key and item were never in the same context window

# Now we need to put this into a list of lists:
# [centre_word, other_word, 0 or 1]
train_df = create_dataset(
    data=train_combinations,
    k=num_negatives,
    num_epochs=num_pos_neg,
    vocab_size=vocab_size,
)
valid_df = create_dataset(
    data=valid_combinations,
    k=num_negatives,
    num_epochs=num_pos_neg,
    vocab_size=vocab_size,
)

# Separate out the pairs data
(
    train_pairs_CL,
    valid_pairs_CL,
    train_labels_CL,
    valid_labels_CL,
) = get_val_train_datasets(train_df=train_df, valid_df=valid_df)

# Create data loaders for our datasets
# https://datascience.stackexchange.com/questions/45916/loading-own-train-data-and-labels-in-dataloader-using-pytorch
# Convert data into PyTorch TensorDataset
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_pairs_CL), torch.tensor(train_labels_CL).float()
)
print(f"Train shape = {torch.tensor(train_pairs_CL).shape}")
valid_dataset = torch.utils.data.TensorDataset(
    torch.tensor(valid_pairs_CL), torch.tensor(valid_labels_CL).float()
)
print(f"Validation shape = {torch.tensor(valid_pairs_CL).shape}")
# Create data loaders
training_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size
)
validation_loader = torch.utils.data.DataLoader(
    valid_dataset, shuffle=True, batch_size=batch_size
)

print(f"Data transformation complete. Time {time.time() - start_time} seconds")

# ------------------------------------------
# Define vanilla skip-gram with negative sampling model
# ------------------------------------------
print(
    torch.cuda.get_device_name(torch.cuda.current_device())
)  # Check if GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model, loss, optimiser, and number of epochs
model = SkipGramNegativeSampling(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    normalise=normalise_embeddings,
).to(device)
binary_cross_entropy = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
start_time = time.time()
with open(f"out/{filename}.txt", "a") as output_file:
    output_file.write("Beginning training loop\n")

for epoch in range(num_epochs):
    running_loss = 0.0
    # num optimisation steps in each epoch
    total_steps = len(training_loader) // accumulation_steps

    for i, data in enumerate(training_loader):
        if i % 500 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(training_loader)}")

        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        output = model(inputs)

        # Compute the loss
        loss = binary_cross_entropy(output, labels)

        # Calculate gradients but don't perform optimisation step
        loss = loss / accumulation_steps  # Normalise loss
        loss.backward()

        # Accumulate gradients over several batches then optimise
        if ((i + 1) % accumulation_steps == 0) or (i == len(training_loader) - 1):
            optimiser.step()
            optimiser.zero_grad()

        # Get the running loss by scaling the loss back to the original value
        running_loss += loss.item() * accumulation_steps

    # Disable gradient computation and reduce memory consumption.
    # Source: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    with torch.no_grad():
        running_vloss = 0.0
        for vi, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = binary_cross_entropy(voutputs, vlabels)
            running_vloss += vloss

    # Print the epoch and average training and validation loss per batch
    print(
        f"Epoch {epoch + 1}/{num_epochs}: Training loss: {running_loss/(i+1)}, Validation loss: {running_vloss.item()/(vi+1)}"
    )
    with open(f"out/{filename}.txt", "a") as output_file:
        output_file.write(
            f"\nEpoch {epoch + 1}/{num_epochs}: Training loss: {running_loss/(i+1)}, Validation loss: {running_vloss.item()/(vi+1)}"
        )
        output_file.write(f"\nTime minutes = {(time.time()-start_time)/60.0}")

    # Check for early stopping
    if running_vloss.item() / (vi + 1) < best_validation_loss:
        best_validation_loss = running_vloss.item() / (vi + 1)
        num_epochs_no_improvement = 0
    else:
        num_epochs_no_improvement += 1

    if num_epochs_no_improvement >= patience:
        with open(f"out/{filename}.txt", "a") as output_file:
            output_file.write(f"\nEarly stopping triggered\n")
        break

# After training, save the trained model's parameters
center_embeddings = model.embedding_centre.weight.data.cpu().numpy()
# Save the center embeddings as a NumPy array
with open(f"out/embeddings_{config}.npy", "wb") as f:
    np.save(f, center_embeddings)
