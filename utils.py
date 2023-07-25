"""A module which stores helper functions used in the main codebase."""
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

# ------------------------------------------
# Data processing
# ------------------------------------------


def data_preprocessing(txt_list: list) -> list:
    """Text normalisation and pre-processing.

    Args:
        txt_list (list): Dataset to process.

    Returns:
        list: List of strings in standardised format.
    """
    # Numbers go to zero
    processed_strings = [re.sub(r"\d", "0", string) for string in txt_list]

    # Replace / and \ with spaces: I don"t want these to become empty strings
    # with the other punctuation because many words get combined into a long word
    # - see GPG/OpenPGP key\fingerpring in test set
    # Same with hyphen: farm-related = farm related and not farmrelated
    processed_strings = [re.sub("/", " ", string) for string in processed_strings]
    processed_strings = [re.sub(r"\\", " ", string) for string in processed_strings]
    processed_strings = [re.sub("-", " ", string) for string in processed_strings]

    # Remove newlines
    processed_strings = [re.sub("\n", "", string) for string in processed_strings]

    # Remove punctuation
    processed_strings = [re.sub(r"[^\w\s]", "", string) for string in processed_strings]

    # Convert multiple zeros to a single zero
    processed_strings = [re.sub(r"0+", " 0 ", string) for string in processed_strings]

    # Replace all capitals
    processed_strings = [string.lower() for string in processed_strings]

    # Remove multiple spaces
    processed_strings = [re.sub(r"\s+", " ", string) for string in processed_strings]

    # Remove leading and trailing whitespace
    processed_strings = [string.strip() for string in processed_strings]

    # Break strings apart
    processed_strings = [string.split() for string in processed_strings]

    # Strip out fluffly words and stop words
    fluff = [
        "the",
        "is",
        "were",
        "was",
        "and",
        "a",
        "an",
        "that",
        "in",
        "at",
        "to",
        "on",
        "are",
        "by",
        "has",
        "had",
    ]
    processed_strings = [
        [word for word in sentence if word not in fluff]
        for sentence in processed_strings
    ]

    return processed_strings


# ------------------------------------------
# Convert text data to numbers
# ------------------------------------------


def text_2_num(txt_list: list[str]) -> dict:
    """Convert text data to numerical representation and provide the lookup dictionary.

    Args:
        txt_list (list[str]): Input data
                              e.g. ["< blah blah blah >", "< example text >", ...]

    Returns:
        dict: Lookup dictionary.
    """
    # Storage dictionary
    text_to_number = {}

    # Build the dictionary
    cnt = 0
    for instance in txt_list:
        for string in instance:
            # Check if the string does not exist in the dictionary
            if string not in text_to_number:
                text_to_number[string] = cnt
                # Increment the string key
                cnt += 1
    # Include the <UNK> key
    text_to_number["<UNK>"] = cnt
    print(f"Token of unknown string = {cnt}")

    return text_to_number


def convert_text_2_number(txt_list: list[str], lookup_dict: dict) -> list[list[int]]:
    """Convert the text data to numerical data including unknown words.

    Args:
        txt_list (list[str]): Input data
                              e.g. ["< blah blah blah >", "< example text >", ...]
        lookup_dict (dict): Dictionary with strings as keys and numbers as items.

    Returns:
        list[list[int]]: Converted data e.g. [[1, 3, 3, 3, 100],[1, 2, 5, 100],...]
    """
    # When dictionary is built, convert the data in the txt_list to numerical format
    number_list = []
    for instance in txt_list:
        # Replace the strings/words not in lookup dictionary with "<UNK>"
        unk_list = [word if word in lookup_dict else "<UNK>" for word in instance]
        # Lookup the numerical representation of the word
        numerical_instance = [lookup_dict[string] for string in unk_list]
        number_list.append(numerical_instance)

    return number_list


# ------------------------------------------
# Data setup for:
# Vanilla skip-gram with negative sampling
# ------------------------------------------


# Break data into context windows
def context_window(data: list[list[int]], window: int) -> dict:
    """Obtain data in the form of a dictionary where keys = centre words
    and items = list of context words for keys (allowing repetition).

    Args:
        data (list[list[int]]): Input data in numerical form.
        window (int): Window for getting paired context + centre words.

    Returns:
        dict: Lookup dictionary for centre words (looking up positive pairs).
    """
    # Create a dictionary where the centre number is the key
    # The item is a list of the words occuring in the windows of context words.
    dictionary_centre_word = {}
    # Middle key
    middle = (window - 1) // 2
    for sentence in data:
        for cnt, word in enumerate(sentence):
            # Only consider ranges where you can get a "middle"
            if (window <= len(sentence)) and (middle <= cnt < len(sentence) - middle):
                # Initialise word as dictionary key if it doesn"t exist
                if word not in dictionary_centre_word:
                    dictionary_centre_word[word] = []
                # Add words before middle
                for j in range(middle, 0, -1):
                    dictionary_centre_word[word].append(sentence[cnt - j])
                # Add words after middle
                for j in range(1, middle + 1, 1):
                    dictionary_centre_word[word].append(sentence[cnt + j])

    return dictionary_centre_word


def create_dataset(
    data: dict, k: int, num_epochs: int, vocab_size: int
) -> list[list[int]]:
    """Create the dataset in an offline fashion.

    Args:
        data (dict): Lookup dictionary for centre words (actual positive pairs, with repeats).
        k (int): Number of negatives to sample per positive instance.
        num_epochs (int): Number of epochs of data - i.e. the number of times k values are sampled per positive instance.
        vocab_size (int): Number of words in vocab, including <UNK>

    Returns:
        list[list[int]]: List of pairs and label of format [centre_word, other_word, 0 or 1].
    """
    # Storage
    data_all_epochs = []
    for epoch in range(num_epochs):
        # Create a list of lists:
        # Each sublist has three elements:
        # 1. Centre word of the window
        # 2. Positive or negative instance
        # 3. 0 or 1 - positive instance = 1 & negative = 0
        paired_lists = []

        for key, item_list in data.items():
            # First do all positives - we will shuffle later
            for item in item_list:
                paired_lists.append([key, item, 1])
            # Now we sample negatives
            sampled_negatives = 0
            while sampled_negatives < k:
                # Make a random guess and see if it is not in positive list
                # Going through all possible negatives took impossibly long
                guess = np.random.choice(a=vocab_size, size=1)[0]
                if (guess not in item_list) and (guess != key):
                    paired_lists.append([key, guess, 0])
                    sampled_negatives += 1
        # Now shuffle paired list
        random.shuffle(paired_lists)
        # Append
        data_all_epochs.append(paired_lists)

        print(f"Epoch {epoch+1} processing complete :)")

    # Flatten the top level of the list (epochs)
    flatten_data_all_epochs = [
        item for epoch_list in data_all_epochs for item in epoch_list
    ]

    return flatten_data_all_epochs


def get_val_train_datasets(train_df: list[list[int]], valid_df: list[list[int]]):
    """Seperate out pairs and labels from training and validation data.

    Args:
        train_df (list[list[int]]): Training data after create_dataset function.
        valid_df (list[list[int]]): Validation data after create_dataset function.

    Returns:
        _type_: pairs and labels from training and validation data
    """
    train_pairs_CL = [triplet[:2] for triplet in train_df]  # [[1,2],[6,3],[7,9],...]
    valid_pairs_CL = [triplet[:2] for triplet in valid_df]
    # Separate out the labels (0 or 1)
    train_labels_CL = [triplet[-1] for triplet in train_df]  # [0,1,1,...]
    valid_labels_CL = [triplet[-1] for triplet in valid_df]

    return train_pairs_CL, valid_pairs_CL, train_labels_CL, valid_labels_CL


# ------------------------------------------
# First try vanilla skip-gram with negative sampling
# ------------------------------------------


class SkipGramNegativeSampling(nn.Module):
    """Skip-gram word embedding model with negative sampling."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, normalise: bool
    ) -> None:
        """Initialisation of the SkipGramNegativeSampling class.

        Args:
            num_embeddings (int): Number of embeddings/ vocabulary size including <UNK>.
            embedding_dim (int): Dimension of the embedding/ latent representation per word.
        """
        super(SkipGramNegativeSampling, self).__init__()

        # Embedding layers
        # num_embeddings = vocabulary size including <UNK>
        self.embedding_centre = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_context = nn.Embedding(num_embeddings, embedding_dim)

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

        # Normalise embeddings or not
        self.normalise = normalise

    def forward(self, paired_input: torch.Tensor) -> torch.Tensor:
        """This function performs a forward pass through the embedding model.

        Args:
            paired_input (torch.Tensor): Paired input where the first column is `centre_input` and the second column is `other_input`.
                - `centre_input` is the centre word from a context window
                - `other_input` is either:
                    - a randomly sampled negative word that does not fall in the same context window as the centre word ever.
                    - a positive instance (a word falling in the same context window as the centre word).

        Returns:
            torch.Tensor: Sigmoid output to be used in the binary cross entropy loss function.
        """
        # Seperate out inputs
        centre_input, other_input = paired_input[:, 0], paired_input[:, 1]

        # Pass inputs through the embedding layer
        embedded_centre = self.embedding_centre(
            centre_input
        )  # We want to use this embedding layer after training
        embedded_other = self.embedding_context(other_input)

        if self.normalise:
            # Perform l2 normalisation
            embedded_centre = torch.nn.functional.normalize(embedded_centre, p=2, dim=1)
            embedded_other = torch.nn.functional.normalize(embedded_other, p=2, dim=1)

        # Compute dot product
        dot_product = torch.sum(embedded_centre * embedded_other, dim=1)

        # Apply sigmoid activation function
        output = self.sigmoid(dot_product)

        return output


# ------------------------------------------
# Visualise embeddings
# ------------------------------------------


def visualise_embeddings(
    train_dictionary: dict,
    embeddings: np.ndarray,
    save_location: str,
    save_name: str,
    perp: float,
) -> None:
    """Visualise the embeddings in two dimensions (reduced substantially) from the trained model using t-SNE.

    Args:
        train_dictionary (dict): Dictionary of words:numbers.
        embeddings (np.ndarray): Saved, trained embeddings
        save_location (str): Folder location to save image to.
        save_name (str): Name of file for saving.
        perp (float): Perplexity for t-SNE algorithm.
    """
    # Highlight specific words
    bus_words = [
        "sales",
        "work",
        "market",
        "finance",
        "stock",
        "trade",
        "economic",
        "interest",
        "bank",
        "shares",
    ]
    sci_tech_words = [
        "software",
        "encryption",
        "computer",
        "cybersecurity",
        "security",
        "linux",
        "engineer",
        "internet",
        "developers",
        "java",
    ]
    sport_words = [
        "soccer",
        "match",
        "olympic",
        "team",
        "football",
        "tournament",
        "coach",
        "cup",
        "olympics",
        "overtime",
    ]
    world_words = [
        "prime",
        "minister",
        "washington",
        "government",
        "federal",
        "law",
        "foreign",
        "president",
        "elections",
        "vote",
    ]
    # Convert to numerical representations
    bus_nums = [train_dictionary[string] for string in bus_words]
    sci_tech_nums = [train_dictionary[string] for string in sci_tech_words]
    sport_nums = [train_dictionary[string] for string in sport_words]
    world_nums = [train_dictionary[string] for string in world_words]
    # Extract the embeddings of the selected words
    selected_words_embeddings = embeddings[
        np.array(bus_nums + sci_tech_nums + sport_nums + world_nums)
    ]

    # Build and fit t-SNE model
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    tsne = TSNE(n_components=2, perplexity=perp)
    sample_embeddings = tsne.fit_transform(selected_words_embeddings)
    # Build figure
    plt.figure(figsize=(10, 8))
    idx = [
        len(bus_words),
        len(bus_words) + len(sci_tech_words),
        len(bus_words) + len(sci_tech_words) + len(sport_words),
    ]

    plt.scatter(
        sample_embeddings[: idx[0], 0],
        sample_embeddings[: idx[0], 1],
        color="blue",
        label="Business",
    )
    plt.scatter(
        sample_embeddings[idx[0] : idx[1], 0],
        sample_embeddings[idx[0] : idx[1], 1],
        color="red",
        label="Sci-Tech",
    )
    plt.scatter(
        sample_embeddings[idx[1] : idx[2], 0],
        sample_embeddings[idx[1] : idx[2], 1],
        color="green",
        label="Sports",
    )
    plt.scatter(
        sample_embeddings[idx[2] :, 0],
        sample_embeddings[idx[2] :, 1],
        color="purple",
        label="World",
    )
    # Provide labels for the embeddings
    for i, word in enumerate(bus_words + sci_tech_words + sport_words + world_words):
        plt.annotate(
            word,
            (sample_embeddings[i, 0] + 0.02, sample_embeddings[i, 1] + 0.02),
            fontsize=13,
        )
    # Legend
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.9), fontsize=14)
    # Remove some borders (top and right)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    # Save figure
    plt.savefig(f"{save_location}/{save_name}.png")
    plt.clf()


# Print distances between a subset of relevant words
def compute_distances(
    train_dictionary: dict, embeddings: np.ndarray, save_location: str, save_name: str
) -> None:
    """Compute Euclidean and cosine distances between the embeddings of a selected set of words.

    Visualise the distance matrices, and save raw matrices.

    Args:
        train_dictionary (dict): Dictionary of words:numbers.
        embeddings (np.ndarray): Saved embeddings from training.
        save_location (str): Folder name to save matrix and image to
        save_name (str): Name of configuration to identify which model this belongs to.
    """
    # List of words to consider
    word_list = [
        "sales",
        "work",
        "market",
        "finance",
        "stock",
        "trade",
        "economic",
        "interest",
        "bank",
        "shares",
        "loan",
        "invest",
        "economy",
        "cash",
        "growth",
        "inflation",
        # 16 business words
        "software",
        "encryption",
        "computer",
        "spyware",
        "cybersecurity",
        "security",
        "linux",
        "engineer",
        "app",
        "internet",
        "web",
        "code",
        "system",
        "server",
        "developers",
        "java",
        # 16 sci-tech words
        "soccer",
        "match",
        "olympic",
        "team",
        "football",
        "tournament",
        "coach",
        "cup",
        "semifinals",
        "win",
        "season",
        "finals",
        "medal",
        "olympics",
        "overtime",
        "goal",
        # 16 sports words
        "prime",
        "minister",
        "authoritarian",
        "washington",
        "government",
        "federal",
        "law",
        "foreign",
        "president",
        "elections",
        "judge",
        "vote",
        "leader",
        "officials",
        "political",
        "politics"
        # 16 world-news words
    ]
    # Number of words
    num_words = len(word_list)

    # Convert word_set to numerical representations
    word_nums = [train_dictionary[word] for word in word_list]

    # Extract the embeddings of the selected words
    selected_words_embeddings = embeddings[np.array(word_nums)]

    # Now we need to find distances
    # 1. Euclidean distance
    # 2. Cosine similarity
    euclid_dist = np.zeros((num_words, num_words))
    cosine_dist = np.zeros((num_words, num_words))

    for row in range(num_words):
        for col in range(num_words):
            # Euclidean distance
            euclid_dist[row, col] = np.linalg.norm(
                selected_words_embeddings[row, :] - selected_words_embeddings[col, :]
            )
            # Cosine distance
            cosine_dist[row, col] = np.dot(
                selected_words_embeddings[row, :], selected_words_embeddings[col, :]
            ) / (
                np.linalg.norm(selected_words_embeddings[row, :])
                * np.linalg.norm(selected_words_embeddings[col, :])
            )

    # Create a 4x4 array to get inter-class similarity
    avg_distances_array_euclid = np.zeros((4, 4))
    avg_distances_array_cosine = np.zeros((4, 4))
    mapping_dict = {
        "euclid": {"dist": euclid_dist, "avg_dist": avg_distances_array_euclid},
        "cosine": {"dist": cosine_dist, "avg_dist": avg_distances_array_cosine},
    }

    idxes = [0, 16, 32, 48, 64]
    for key in mapping_dict:
        arr = mapping_dict[key]["dist"]
        total_diag = 240 # 16 x 16 - 16
        # Inter-class similarity
        mapping_dict[key]["avg_dist"][0, 0] = (
            np.sum(arr[idxes[0] : idxes[1], idxes[0] : idxes[1]])
            - np.sum(np.diag(arr[idxes[0] : idxes[1], idxes[0] : idxes[1]]))
        ) / total_diag
        mapping_dict[key]["avg_dist"][1, 1] = (
            np.sum(arr[idxes[1] : idxes[2], idxes[1] : idxes[2]])
            - np.sum(np.diag(arr[idxes[1] : idxes[2], idxes[1] : idxes[2]]))
        ) / total_diag
        mapping_dict[key]["avg_dist"][2, 2] = (
            np.sum(arr[idxes[2] : idxes[3], idxes[2] : idxes[3]])
            - np.sum(np.diag(arr[idxes[2] : idxes[3], idxes[2] : idxes[3]]))
        ) / total_diag
        mapping_dict[key]["avg_dist"][3, 3] = (
            np.sum(arr[idxes[3] :, idxes[3] :])
            - np.sum(np.diag(arr[idxes[3] :, idxes[3] :]))
        ) / total_diag
        # Cross similarity
        # Business and ...
        mapping_dict[key]["avg_dist"][0, 1] = mapping_dict[key]["avg_dist"][
            1, 0
        ] = np.mean(arr[idxes[0] : idxes[1], idxes[1] : idxes[2]])
        mapping_dict[key]["avg_dist"][0, 2] = mapping_dict[key]["avg_dist"][
            2, 0
        ] = np.mean(arr[idxes[0] : idxes[1], idxes[2] : idxes[3]])
        mapping_dict[key]["avg_dist"][0, 3] = mapping_dict[key]["avg_dist"][
            3, 0
        ] = np.mean(arr[idxes[0] : idxes[1], idxes[3] :])
        # Sci-Tech and ...
        mapping_dict[key]["avg_dist"][1, 2] = mapping_dict[key]["avg_dist"][
            2, 1
        ] = np.mean(arr[idxes[1] : idxes[2], idxes[2] : idxes[3]])
        mapping_dict[key]["avg_dist"][1, 3] = mapping_dict[key]["avg_dist"][
            3, 1
        ] = np.mean(arr[idxes[1] : idxes[2], idxes[3] :])
        # Sports and ...
        mapping_dict[key]["avg_dist"][2, 3] = mapping_dict[key]["avg_dist"][
            3, 2
        ] = np.mean(arr[idxes[2] : idxes[3], idxes[3] :])

    # Save figure of matrix
    class_labels = ["Business", "Sci/Tech", "Sports", "World"]

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        mapping_dict["euclid"]["avg_dist"],
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        cbar=False,
    )
    # Set the tick labels for both x-axis and y-axis
    plt.xticks(
        np.arange(len(class_labels)) + 0.5, class_labels, rotation=45, ha="right"
    )
    plt.yticks(
        np.arange(len(class_labels)) + 0.5, class_labels, rotation=0, va="center"
    )
    plt.savefig(f"{save_location}/{save_name}_avg_distances_euclid.png")
    plt.clf()

    # Cosine
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        mapping_dict["cosine"]["avg_dist"],
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        cbar=False,
    )
    # Set the tick labels for both x-axis and y-axis
    plt.xticks(
        np.arange(len(class_labels)) + 0.5, class_labels, rotation=45, ha="right"
    )
    plt.yticks(
        np.arange(len(class_labels)) + 0.5, class_labels, rotation=0, va="center"
    )
    plt.savefig(f"{save_location}/{save_name}_avg_distances_cosine.png")
    plt.clf()


# ------------------------------------------
# Text classification
# ------------------------------------------


# Convert labels to numerical representation
def labels_numerical(
    labels: list[str],
    categories: dict = {"Sports": 0, "World": 1, "Sci/Tech": 2, "Business": 3},
) -> np.ndarray:
    """Convert word labels to a one-hot encoded numpy array.

    Args:
        labels (list[str]): List of categorical labels.
        categories (dict, optional): A dictionary encoding the categorical labels to numbers, starting at 0. Defaults to {"Sports": 0, "World": 1, "Sci/Tech": 2, "Business": 3}.

    Returns:
        np.ndarray: One-hot encoded numpy array
    """
    # Convert categories to numbers
    numerical_labels = [categories[word] for word in labels]
    numerical_labels = np.array(numerical_labels)

    # Convert to one-hot matrix where rows = instances and columns are classes (specified by categories)
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
    one_hot_matrix = np.zeros((numerical_labels.size, numerical_labels.max() + 1))
    one_hot_matrix[np.arange(numerical_labels.size), numerical_labels] = 1

    return one_hot_matrix


# Find average sentence embeddings
def average_embedding(sentence: list[np.ndarray]) -> np.ndarray:
    """Get the average sentence embedding given a list of word embeddings.

    Args:
        sentence (list[np.ndarray]): A list of word embeddings from a sentence.

    Returns:
        np.ndarray: Average embedding of the sentence.
    """
    return np.mean(sentence, axis=0)


# Convert list[list[int]] to list[list[np.ndarray]] to list[np.ndarray]
def convert_data(data: list[list[int]], embedding: np.ndarray) -> list[np.ndarray]:
    """Convert data from form list[list[int]] to list[np.ndarray].

    Each sentence (e.g. [2,3,7,4]) is replaced by the average word embedding from the words in the sentence.

    Args:
        data (list[list[int]]): Input data after text converted to numbers.
        embedding (np.ndarray): Trained embeddings.

    Returns:
        list[np.ndarray]: List of average sentence embeddings.
    """
    sentence_embeddings = []
    for sentence in data:
        # list[int] -> list[np.ndarray]
        sentence_embedding = []
        for word in sentence:
            # Get embedding
            sentence_embedding.append(embedding[word, :])
        # Get the average sentence embedding
        sentence_embeddings.append(average_embedding(sentence_embedding))
    return sentence_embeddings


def to_TensorDataset(dat: dict) -> torch.utils.data.TensorDataset:
    """Get data from dictionary to TensorDataset form.

    Args:
        dat (dict): Dictionary with string keys "sentence_embeddings" and "numerical_labels"

    Returns:
        torch.utils.data.TensorDataset: _description_
    """
    return torch.utils.data.TensorDataset(
        torch.tensor(dat["sentence_embeddings"]),
        torch.tensor(dat["numerical_labels"]).float(),
    )


# Text classifier
class SentenceMLP(nn.Module):
    def __init__(self, input_size: int, nodes_per_layer: list[int], num_classes: int):
        """Initialise the text classifier.

        Args:
            input_size (int): Embedding dimension.
            nodes_per_layer (list[int]): A list of the number of dense nodes per layer.
            num_classes (int): Number of classes for classification.
        """
        super(SentenceMLP, self).__init__()
        # Create a list of all the hidden layers (including input layer)
        layers = []
        prev_layer_nodes = input_size
        for layer_nodes in nodes_per_layer:
            # Dense layer
            layers.append(nn.Linear(prev_layer_nodes, layer_nodes))
            # Relu activation function
            layers.append(nn.ReLU())
            prev_layer_nodes = layer_nodes
        # Build a sequential model with all the hidden (and input) layers
        self.hidden_layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(prev_layer_nodes, num_classes)

    def forward(self, input_sentence: torch.Tensor) -> torch.Tensor:
        """A single forward pass through the model.

        Args:
            input_sentence (torch.Tensor): A batch of average sentence embeddings.

        Returns:
            torch.Tensor: The output of a softmax layer to be used in the loss function.
        """
        # forward pass through the model
        x = self.hidden_layers(input_sentence)
        x = self.out_layer(x)
        out = nn.functional.softmax(x, dim=1)
        return out


# Plot the confusion matrix as a heatmap
def test_confusion_matrix(
    conf_matrix: np.ndarray, class_labels: list, save_location: str, save_name: str
) -> None:
    """Save the test confusion matrix as a png.

    Args:
        conf_matrix (np.ndarray): Test confusion matrix
        class_labels (list): List of class labels corresponding to one-hot encodings.
        save_location (str): Folder to save png in.
        save_name (str): Name of saved png.
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        cmap="coolwarm",
        fmt="d",
        cbar=False,
    )
    # Set the tick labels for both x-axis and y-axis
    plt.xticks(
        np.arange(len(class_labels)) + 0.5, class_labels, rotation=45, ha="right"
    )
    plt.yticks(
        np.arange(len(class_labels)) + 0.5, class_labels, rotation=0, va="center"
    )
    plt.savefig(f"{save_location}/{save_name}_confusion.png")
    plt.clf()


# ------------------------------------------
# STS task
# ------------------------------------------


def filter_word_pairs(word_sim_df: pd.DataFrame, word_token_dict: dict):
    """Filter out word pairs not present in training data for STS task.

    Args:
        word_sim_df (pd.DataFrame): WordSim dataset
        (https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd)
        word_token_dict (dict): Words to tokens from training data

    Returns:
        tuple(list, list): word pairs and human annotated perception of similarity
        + word pairs for evaluation by trained word embedding model

    """
    filtered_pairs = []  # actual scores
    eval_pairs = []

    # Iterate through dataset and filter out rows for which we have no data
    for _, row in word_sim_df.iterrows():
        word1, word2, score = row["Word 1"], row["Word 2"], row["Human (Mean)"]

        # Check if both words exist in the word_token_dict
        if word1 in word_token_dict and word2 in word_token_dict:
            # Get the token values
            token1, token2 = word_token_dict[word1], word_token_dict[word2]
            filtered_pairs.append([token1, token2, score])
            eval_pairs.append([token1, token2])

    return filtered_pairs, eval_pairs


def compute_similarity_scores(eval_pairs: list, embedding_matrix: np.ndarray) -> list:
    """Determine the semantic textual similarity scores using a trained word embedding matrix.

    Args:
        eval_pairs (list): Pairs of words/tokens for evaluation.
        embedding_matrix (np.ndarray): Trained embedding matrix

    Returns:
        list: List of tokens and cosine similarity from the trained model.
    """
    similarity_scores = []
    for tokens in eval_pairs:
        # Compute cosine similarity
        similarity_score = np.dot(
            embedding_matrix[tokens[0], :], embedding_matrix[tokens[1], :]
        ) / (
            np.linalg.norm(embedding_matrix[tokens[0], :])
            * np.linalg.norm(embedding_matrix[tokens[1], :])
        )
        similarity_scores.append([tokens[0], tokens[1], similarity_score])

    return similarity_scores
