# YouTube Comment Sentiment & Topic Analysis

This project analyzes YouTube comments using sentiment analysis and topic modeling.

The app allows users to enter a YouTube video URL or use a default video URL. 

It extracts public YouTube comments using the YouTube Data API and analyzes the comments using multiple NLP models and visualization techniques.

> Note: This project is a recreation of a group project originally developed as part of graduate-level data analytics coursework. It has been adapted into a portfolio project that can analyze different types of YT videos, demonstrating applications of NLP, sentiment analysis,and topic modeling.

## Project Overview

This project analyzes public YouTube comments to understand audience sentiment and discussion topics.

The goal of this project is to create a reusable NLP dashboard that can analyze comments from different YouTube videos rather than being limited to one fixed dataset.

## Tools

Python, Streamlit, YouTube Data API, pandas, NumPy, Plotly, Matplotlib, WordCloud, VADER, Hugging Face Transformers DistilBERT, RoBERTa, spaCy, Gensim

## Methodology

### 1. Data Collection

First, comments are collected from YouTube using the YouTube Data API.

Using provided YouTube video URL, Video ID is extracted from the URL and then used to request comment data from the API.

### 2. Data Cleaning

After extracting comments, the raw text data is cleaned.

The cleaning function removes unnecessary formatting such as line breaks, URLs, extra spaces, and special characters. Empty comments and duplicate comments are removed before analysis.

This step helps improve the quality of the sentiment analysis and topic modeling results.

### 3. Sentiment Model Processing

The cleaned comments are passed into three sentiment models.

VADER generates a compound sentiment score and classifies each comment as positive, negative, or neutral based on score thresholds.

DistilBERT and RoBERTa use Hugging Face transformer pipelines to classify each comment and return a sentiment label with a confidence score.

The results from all three models are combined into one final results table.

### 4. Model Comparison

After all sentiment models are applied, a comparison table is created.

The table counts how many comments were classified into each sentiment category by each model.

### 5. NLP Preprocessing

For word frequency analysis and topic modeling, the app performs additional NLP preprocessing.

The comments are tokenized and lemmatized using spaCy. Stop words, punctuation, URLs, emails, and very short tokens are removed.

This produces a cleaned token list for each comment.

### 6. Word Frequency Visualization

All cleaned tokens are combined into a single word list.

The app counts word frequency using Python’s Counter function.

The word frequency results are used to create:

- A word cloud
- A treemap

### 7. Topic Modeling

Multiple NLP techniques are used to identify common discussion patterns in the YouTube comments.

For topic modeling, the cleaned tokens are converted into a dictionary and corpus using Gensim. Then Latent Dirichlet Allocation, also known as LDA, is applied to group comments into a selected number of topics based on word patterns. The model returns the top keywords for each topic, and a readable topic name using the most important keywords.

In addition to LDA, N-gram analysis is also used to extract common phrases from the comments. This includes frequently repeated three-word phrases, which can help capture meaning that may not be clear from single words alone.

TF-IDF keyword extraction is used to identify important words in the comment dataset. TF-IDF gives more weight to words that are meaningful within the dataset and less weight to very common words that appear across many comments.


### 8. Topic and Keyword Score Assignment

For LDA analysis, model returns topic probabilites and highest probabilities are selected as the dominant topic

For N-gram analysis, the most frequently occurring phrases are identified across the cleaned comments.

For TF-IDF analysis, keyword importance scores are calculated. Higher TF-IDF scores indicate that a word is more important or distinctive within the comment dataset.

## App Features

### 1. YouTube Comment Extraction

The app uses the YouTube Data API to extract public top-level comments from a YouTube video.

Users can enter a YouTube video URL directly in the app. A default video URL is also provided in the input field so the app can be tested immediately.

The extracted data includes:

- Video ID
- Comment date
- Comment text

### 2. Text Cleaning and Preprocessing

Before running sentiment analysis and topic modeling, the raw comments are cleaned and prepared.

The cleaning process includes:

- Removing line breaks
- Removing URLs
- Removing extra spaces
- Removing duplicate comments
- Standardizing text format

For topic modeling, the comments are further processed using tokenization and lemmatization. Stop words, punctuation, URLs, and short tokens are removed to create cleaner text features.

### 3. Sentiment Analysis

The app compares results from three sentiment models:

| Model | Description |
|---|---|
| VADER | Rule-based sentiment model commonly used for short social media-style text |
| DistilBERT | Transformer-based sentiment model trained for English sentiment classification |
| RoBERTa | Transformer-based model designed for social media-style sentiment analysis |

Each model assigns a sentiment label to every comment such as positive, negative or neutral.

The app displays both individual model results and a comparison across all three models.

### 4. Sentiment Model Comparison

The app creates a comparison table showing how many comments each model classified as positive, negative, or neutral.

It also includes a grouped bar chart to visually compare the sentiment distribution across VADER, DistilBERT, and RoBERTa.

This helps show that different sentiment models may interpret the same comment differently because they use different methods.

### 5. Individual Sentiment Results

Each sentiment model has its own result section.

For each model, the app displays

- Sentiment distribution chart
- Top positive comments
- Top negative comments
- Sentiment score

This allows inspection of not only the overall results, but also the actual comments behind the predictions.

### 6. Word Frequency Analysis

The app performs word frequency analysis on cleaned and lemmatized comments.

This section includes:

- Word cloud
- Word frequency treemap

The word cloud gives a quick visual summary of commonly used words in the comment section.

The treemap shows the most frequent words in a structured chart where larger boxes represent more frequent words.

### 7. Topic Modeling 

The app uses Latent Dirichlet Allocation, also known as LDA, to identify common discussion topics in the YouTube comments.

In addition to LDA, the app also extracts N-grams to identify common three-word phrases in the comments. 

The app also uses TF-IDF keyword extraction to identify important words that appear frequently in specific comments or groups of comments.

### 8. Downloadable Results

The app allows users to download analysis results as CSV files.

Available downloads include:

- Sentiment analysis results
- Topic modeling results

This makes it easier to review the results outside the app or use them for further analysis.

## Conclusion and Future Work

### Conclusion

Based on the sample YouTube videos tested so far, RoBERTa appears to be the most suitable model for analyzing sentiment in YouTube comments. Compared with the other sentiment models, RoBERTa is better at capturing the tone and context of short, informal comments, which are common on YouTube.

For topic modeling, the results were less effective. Since YouTube comments are often very short, informal, and lack enough context, the topic modeling output did not always generate clear or meaningful topic labels. Based on the current testing, traditional topic modeling may not be the best approach for labeling themes in short form comments without further improvement.

### Future Work

Future improvements could include adding like counts as an additional component in the analysis. Comments with higher likes may represent opinions that resonate more strongly with viewers, so including this information could provide a better understanding of audience sentiment and reaction.

Another area for future work is to explore more advanced topic modeling techniques that may perform better on short text data. This could include experimenting with transformer-based topic modeling methods or clustering approaches that use sentence embeddings to capture meaning beyond individual words.

## Live Demo

A Streamlit demo is available for testing.

🔗 **App:** [Youtube comments sentiment analysis App](https://sentimentanalysisonyoutube.streamlit.app)

> Note: The Streamlit app may go into sleep mode after a period of inactivity. If the app does not load immediately, please allow a few moments for it to wake up.

