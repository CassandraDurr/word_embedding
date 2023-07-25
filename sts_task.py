"""A module used to assess word embeddings using semantic textual similarity, using WordSim353 as the dataset to compare to."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import (
    compute_similarity_scores,
    data_preprocessing,
    filter_word_pairs,
    text_2_num,
)

# Retrieve train dictionary
with open("data/train_texts.txt", "r") as file:
    train_txt = file.readlines()
train = data_preprocessing(train_txt)
train_dictionary = text_2_num(train)

# Retrieve embeddings
with open("out/embeddings_5_5_True.npy", "rb") as f:
    embeddings = np.load(f)
vocab_size, embedding_dim = embeddings.shape
config = "5_5_True"
print(f"{config}")

# Get WordSim dataset
wordsim_df = pd.read_csv("data/wordsim353crowd.csv")
wordsim_df_filtered, evaluation_pairs = filter_word_pairs(
    word_sim_df=wordsim_df, word_token_dict=train_dictionary
)
print(len(wordsim_df_filtered))

# Get cosine similarities from embeddings
sim_scores = compute_similarity_scores(
    embedding_matrix=embeddings, eval_pairs=evaluation_pairs
)

# Calculate the Spearman rank correlation between human scores and model scores
sim_scores_only = [sublist[-1] for sublist in sim_scores]
wordsim_scores_only = [sublist[-1] for sublist in wordsim_df_filtered]
spearman_corr, p_value = spearmanr(sim_scores_only, wordsim_scores_only)

print(f"Spearman Rank Correlation: {spearman_corr}, p value: {p_value}")
# p > 0.05: correlation is not statistically significant -> may be due to random chance

# 5_5_False
# Spearman Rank Correlation: -0.04336684064043529, p value: 0.43802729971376364
# 5_5_True
# Spearman Rank Correlation: -0.043915373930810395, p value: 0.43224982080047414