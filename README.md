# Assessing Word Embedding Evaluation Methodologies

Learning appropriate word embeddings is crucial for a number of tasks in natural language processing, however, there is currently no accepted gold-standard evaluation criteria for assessing word embeddings. The primary focus of this codebase is to assess learned word embeddings using a number of evaluation criteria and then discuss the benefits and drawbacks of the approaches in a report.

## Data
The data used to train the word embedding model and train the sentence classifier is the AG News dataset. Information about this dataset can be found at: http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html and the dataset is freely available for non-commercial use.

## Requirements
If you are using a Python version 3.10.8 and have:
- matplotlib, 
- seaborn, 
- pandas, 
- numpy,
- scipy,
- sklearn,
- and torch 

installed then the code should run fine. If you do have issues running the code, then you can recreate my exact environment using conda, with commands:
- ``conda create --name nlp_env python=3.10.8``
- ``conda activate nlp_env``
- ``pip install -r requirements.txt``

This should ensure no issues in running the code. 

## Learning word embeddings
To learn the word embeddings, navigate to the top-level directory (where this README is), and execute command `python word_embeddings_torch.py`. 

To change:
- the data sources, change lines 33-51.
- the output file storage, change line 24.
- hyperparameters, change lines 141-146.

## Visualise embeddings and sample key cosine distances
To visualise embeddings and sample key cosine distances, Sections 4.1 and 4.2 from the report, navigate to the top-level directory (where this README is), and execute command `python visualise_embeddings.py`. 

To change:
- the data sources, change lines 11-18.

## Semantic textual similarity (STS)
To perform STS task, Section 4.3 from the report, navigate to the top-level directory (where this README is), and execute command `python sts_task.py`. 

To change:
- the data sources, change lines 14, 20, and 27.

## Text classification
To perform downstream sentence classification, Section 4.4 from the report, navigate to the top-level directory (where this README is), and execute command `python text_classifier.py`. 

To change:
- the data sources, change lines 19-25.
