"""A module used to visualise projected word embeddings and compute distances between a selection of word embeddings."""
import numpy as np

from utils import (
    compute_distances,
    data_preprocessing,
    text_2_num,
    visualise_embeddings,
)

# Load embeddings
with open("out/embeddings_5_5_False.npy", "rb") as f:
    embeddings = np.load(f)
vocab_size, embedding_dim = embeddings.shape
config = "5_5_False"

# Retrieve train dictionary
with open("data/train_texts.txt", "r") as file:
    train_txt = file.readlines()
train = data_preprocessing(train_txt)
train_dictionary = text_2_num(train)

# Visualise
for perp in [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]:
    visualise_embeddings(
        train_dictionary=train_dictionary,
        embeddings=embeddings,
        save_location="out",
        save_name=f"we_{config}_{perp}",
        perp=perp,
    )

# Compute distances between words
compute_distances(
    train_dictionary, embeddings, save_location="out", save_name=f"we_{config}"
)
