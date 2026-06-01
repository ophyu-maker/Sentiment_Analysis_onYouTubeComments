import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from collections import Counter
from itertools import chain

import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers.pipelines import pipeline

import spacy
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models import LdaModel

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="YouTube Comment Sentiment & Topic Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Comment Sentiment & Topic Analysis")
st.write(
    "Analyze YouTube comments using VADER, DistilBERT, RoBERTa, "
    "word frequency, word cloud, treemap, and LDA topic modeling."
)


# =========================================================
# API KEY
# =========================================================

try:
    API_KEY = st.secrets["YOUTUBE_API_KEY"]
except Exception:
    API_KEY = None


# =========================================================
# Helper functions
# =========================================================

def extract_video_id(url_or_id):
    """
    Extract YouTube video ID from URL or return the ID directly.
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - raw VIDEO_ID
    """

    if not url_or_id:
        return None

    text = url_or_id.strip()

    # youtu.be short URL
    match = re.search(r"youtu\.be/([^?&]+)", text)
    if match:
        return match.group(1)

    # standard YouTube URL
    match = re.search(r"v=([^?&]+)", text)
    if match:
        return match.group(1)

    # shorts URL
    match = re.search(r"shorts/([^?&]+)", text)
    if match:
        return match.group(1)

    # assume raw video ID
    return text


def fetch_youtube_comments(video_ids, api_key, max_pages=3):
    """
    Pull top-level YouTube comments for one or multiple video IDs.
    max_pages controls how many API pages to pull per video.
    Each page can return up to 100 comments.
    """

    if isinstance(video_ids, str):
        video_ids = [video_ids]

    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    rows = []

    for vid in video_ids:
        params = {
            "part": "snippet",
            "videoId": vid,
            "maxResults": 100,
            "textFormat": "plainText",
            "key": api_key,
        }

        page_count = 0

        while True:
            response = requests.get(url, params=params)
            data = response.json()

            if "error" in data:
                st.warning(f"Error for video {vid}: {data['error'].get('message')}")
                break

            for item in data.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]

                rows.append({
                    "video_id": vid,
                    "date": snippet.get("publishedAt"),
                    "comment": snippet.get("textDisplay"),
                    "like_count": snippet.get("likeCount"),
                    "author": snippet.get("authorDisplayName")
                })

            page_count += 1

            if "nextPageToken" not in data:
                break

            if max_pages is not None and page_count >= max_pages:
                break

            params["pageToken"] = data["nextPageToken"]

    return pd.DataFrame(rows)


def clean_comment_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def prepare_comments(df):
    df = df.copy()
    df["clean_comment"] = df["comment"].apply(clean_comment_text)
    df = df[df["clean_comment"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_comment"])
    return df.reset_index(drop=True)


# =========================================================
# Cached models
# =========================================================

@st.cache_resource
def load_distilbert_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


@st.cache_resource
def load_roberta_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


# =========================================================
# Sentiment models
# =========================================================

def run_vader(df):
    df_vader = df.copy()
    analyzer = SentimentIntensityAnalyzer()

    df_vader["vader_score"] = df_vader["clean_comment"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    def classify_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df_vader["vader_label"] = df_vader["vader_score"].apply(classify_sentiment)

    return df_vader


def run_distilbert(df):
    df_distilbert = df.copy()
    sentiment_pipeline = load_distilbert_model()

    comments = df_distilbert["clean_comment"].tolist()

    results = sentiment_pipeline(
        comments,
        truncation=True,
        max_length=512,
        batch_size=32
    )

    df_distilbert["distilbert_label"] = [
        result["label"].lower() for result in results
    ]

    df_distilbert["distilbert_score"] = [
        result["score"] for result in results
    ]

    return df_distilbert


def run_roberta(df):
    df_roberta = df.copy()
    sentiment_pipeline = load_roberta_model()

    comments = df_roberta["clean_comment"].tolist()

    results = sentiment_pipeline(
        comments,
        truncation=True,
        max_length=512,
        batch_size=32
    )

    df_roberta["roberta_label"] = [
        result["label"].lower() for result in results
    ]

    df_roberta["roberta_score"] = [
        result["score"] for result in results
    ]

    return df_roberta


def run_all_sentiment_models(df):
    df = prepare_comments(df)

    df_vader = run_vader(df)
    df_distilbert = run_distilbert(df)
    df_roberta = run_roberta(df)

    final_df = df.copy()

    final_df["vader_label"] = df_vader["vader_label"]
    final_df["vader_score"] = df_vader["vader_score"]

    final_df["distilbert_label"] = df_distilbert["distilbert_label"]
    final_df["distilbert_score"] = df_distilbert["distilbert_score"]

    final_df["roberta_label"] = df_roberta["roberta_label"]
    final_df["roberta_score"] = df_roberta["roberta_score"]

    return final_df


def create_model_comparison(results_df):
    comparison_table = pd.DataFrame({
        "VADER": results_df["vader_label"].value_counts(),
        "DistilBERT": results_df["distilbert_label"].value_counts(),
        "RoBERTa": results_df["roberta_label"].value_counts()
    }).fillna(0).astype(int)

    comparison_table = comparison_table.reset_index()
    comparison_table.columns = ["sentiment_label", "VADER", "DistilBERT", "RoBERTa"]

    comparison_long = comparison_table.melt(
        id_vars="sentiment_label",
        var_name="model",
        value_name="comment_count"
    )

    return comparison_table, comparison_long


# =========================================================
# NLP functions
# =========================================================

def lemmatize_comments(df):
    df_nlp = df.copy()
    nlp = load_spacy_model()

    def lemmatize_text(text):
        doc = nlp(text)

        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_url
            and not token.like_email
            and len(token.lemma_) > 2
        ]

        return tokens

    df_nlp["tokens"] = df_nlp["clean_comment"].apply(
        lambda x: simple_preprocess(x, deacc=True)
    )

    df_nlp["token_clean"] = df_nlp["clean_comment"].apply(lemmatize_text)
    df_nlp["token_count"] = df_nlp["token_clean"].apply(len)

    return df_nlp


def get_word_counts(df_nlp):
    all_words = list(chain.from_iterable(df_nlp["token_clean"]))
    word_counter = Counter(all_words)
    word_counts = pd.Series(word_counter.values(), index=word_counter.keys())
    word_counts = word_counts.sort_values(ascending=False)
    return word_counts


def create_wordcloud(word_counts):
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(word_counts)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


def create_treemap(word_counts, top_n=10):
    top_words = word_counts.head(top_n).reset_index()
    top_words.columns = ["word", "count"]

    fig = px.treemap(
        top_words,
        path=["word"],
        values="count"
    )

    return fig


def build_topic_stopwords(user_stopwords_text=""):
    """
    Build reusable topic modeling stopwords.
    Includes:
    - sklearn English stopwords
    - generic YouTube/comment stopwords
    - optional user-entered stopwords
    """

    youtube_stopwords = {
        "video", "watch", "watching", "channel", "comment", "comments",
        "people", "person", "someone", "everyone",
        "really",
        "think", "know", "say", "said", "see",
        "go", "going",
        "one", "thing", "way", "time", "day",
        "would", "could", "should", "also",
        "thank", "thanks", "please"
    }

    user_stopwords = {
        word.strip().lower()
        for word in user_stopwords_text.split(",")
        if word.strip()
    }

    final_stopwords = set(ENGLISH_STOP_WORDS).union(youtube_stopwords).union(user_stopwords)

    return final_stopwords


# =========================================================
# LDA topic modeling
# =========================================================

def run_lda_topic_modeling(df_nlp, num_topics=3, topn=8, user_stopwords_text=""):
    df_topic = df_nlp.copy()

    topic_stopwords = build_topic_stopwords(user_stopwords_text)

    df_topic["token_clean_topic"] = df_topic["token_clean"].apply(
        lambda tokens: [
            word for word in tokens
            if word not in topic_stopwords
            and len(word) > 2
        ]
    )

    df_topic = df_topic[df_topic["token_clean_topic"].apply(len) > 0].reset_index(drop=True)

    dictionary = corpora.Dictionary(df_topic["token_clean_topic"])

    dictionary.filter_extremes(
        no_below=2,
        no_above=0.7
    )

    corpus = [dictionary.doc2bow(tokens) for tokens in df_topic["token_clean_topic"]]

    if len(dictionary) == 0 or len(corpus) == 0:
        return df_topic, pd.DataFrame(), pd.DataFrame()

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=99,
        passes=10,
        alpha="auto",
        per_word_topics=True
    )

    def get_dominant_topic(bow):
        topic_probs = lda_model.get_document_topics(bow)

        if len(topic_probs) == 0:
            return None, 0

        dominant_topic, topic_score = max(topic_probs, key=lambda x: x[1])
        return dominant_topic, topic_score

    dominant_topics = [get_dominant_topic(bow) for bow in corpus]

    df_topic["topic_id"] = [topic[0] for topic in dominant_topics]
    df_topic["topic_score"] = [topic[1] for topic in dominant_topics]

    topic_rows = []

    for topic_id in range(num_topics):
        words = lda_model.show_topic(topic_id, topn=topn)
        keywords = [word for word, weight in words]

        topic_rows.append({
            "topic_id": topic_id,
            "topic_keywords": ", ".join(keywords),
            "topic_name": " / ".join(keywords[:3])
        })

    topic_keywords_df = pd.DataFrame(topic_rows)

    topic_summary = (
        df_topic["topic_id"]
        .value_counts()
        .reset_index()
    )

    topic_summary.columns = ["topic_id", "comment_count"]

    topic_summary = topic_summary.merge(
        topic_keywords_df,
        on="topic_id",
        how="left"
    )

    topic_summary = topic_summary.sort_values("topic_id")

    return df_topic, topic_keywords_df, topic_summary


def prepare_topic_text(df_nlp, user_stopwords_text=""):
    df_topic_text = df_nlp.copy()

    topic_stopwords = build_topic_stopwords(user_stopwords_text)

    df_topic_text["token_clean_topic"] = df_topic_text["token_clean"].apply(
        lambda tokens: [
            word for word in tokens
            if word not in topic_stopwords
            and len(word) > 2
        ]
    )

    df_topic_text["topic_text"] = df_topic_text["token_clean_topic"].apply(
        lambda tokens: " ".join(tokens)
    )

    df_topic_text = df_topic_text[df_topic_text["topic_text"].str.len() > 0]

    return df_topic_text


def run_ngram_analysis(df_nlp, user_stopwords_text="", top_n=20):
    df_topic_text = prepare_topic_text(df_nlp, user_stopwords_text)

    if df_topic_text.empty:
        return pd.DataFrame()

    vectorizer = CountVectorizer(
        ngram_range=(3, 3),
        min_df=2,
        max_df=0.8
    )

    try:
        X = vectorizer.fit_transform(df_topic_text["topic_text"])
    except ValueError:
        return pd.DataFrame()

    ngram_counts = X.sum(axis=0).A1
    ngram_names = vectorizer.get_feature_names_out()

    ngram_df = pd.DataFrame({
        "three_word_phrase": ngram_names,
        "count": ngram_counts
    })

    ngram_df = ngram_df.sort_values("count", ascending=False).head(top_n)

    return ngram_df


def run_tfidf_analysis(df_nlp, user_stopwords_text="", top_n=20):
    df_topic_text = prepare_topic_text(df_nlp, user_stopwords_text)

    if df_topic_text.empty:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    try:
        X = vectorizer.fit_transform(df_topic_text["topic_text"])
    except ValueError:
        return pd.DataFrame()

    tfidf_scores = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame({
        "term": terms,
        "tfidf_score": tfidf_scores
    })

    tfidf_df = tfidf_df.sort_values("tfidf_score", ascending=False).head(top_n)

    return tfidf_df


# =========================================================
# Sidebar
# =========================================================

st.sidebar.header("Settings")


max_pages = st.sidebar.slider(
    "Max comment pages per video",
    min_value=1,
    max_value=4,
    value=2,
    help="Each page can pull up to 100 comments. Higher values may take longer."
)

num_topics = st.sidebar.slider(
    "Number of LDA topics",
    min_value=2,
    max_value=5,
    value=3
)

top_words = st.sidebar.slider(
    "Top words for treemap",
    min_value=10,
    max_value=30,
    value=10
)


# =========================================================
# Input section
# =========================================================

DEFAULT_VIDEO_URLS = "https://www.youtube.com/watch?v=h-l_6617x6A"


st.subheader("Analyze YouTube Video")

user_url = st.text_input(
          "Enter YouTube video URL or Run default video URL",
          value=DEFAULT_VIDEO_URLS
     )

run_button = st.button("Run Analysis", type="primary")


# =========================================================
# Main app logic
# =========================================================

if run_button:

    if API_KEY is None:
        st.error(
            "YouTube API key is missing. Add it to `.streamlit/secrets.toml` as `YOUTUBE_API_KEY`."
        )
        st.stop()

    if not user_url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    video_id = extract_video_id(user_url)

    if not video_id:
        st.warning("Please enter a valid YouTube URL.")
        st.stop()

    with st.spinner("Fetching YouTube comments..."):
        raw_df = fetch_youtube_comments(
            video_ids=video_id,
            api_key=API_KEY,
            max_pages=max_pages
        )

    if raw_df.empty:
        st.error("No comments found. The video may have comments disabled or the API request failed.")
        st.stop()

    st.success(f"Fetched {len(raw_df):,} comments.")

    with st.expander("Preview raw comments"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    # Sentiment analysis runs only when Run Analysis is clicked.
    with st.spinner("Running VADER, DistilBERT, and RoBERTa sentiment models..."):
        results_df = run_all_sentiment_models(raw_df)

    # NLP preprocessing runs once after sentiment analysis.
    with st.spinner("Preparing text for word cloud, treemap, and topic modeling..."):
        df_nlp = lemmatize_comments(results_df)
        word_counts = get_word_counts(df_nlp)

    # Save results so topic modeling can be rerun without rerunning sentiment models.
    st.session_state["results_df"] = results_df
    st.session_state["df_nlp"] = df_nlp
    st.session_state["word_counts"] = word_counts
    st.session_state["latest_df_topic"] = None


if "results_df" not in st.session_state:
    st.info("Choose setting from the sidebar, then click **Run Analysis**.")
    st.stop()

results_df = st.session_state["results_df"]
df_nlp = st.session_state["df_nlp"]
word_counts = st.session_state["word_counts"]


# =====================================================
# KPIs
# =====================================================

st.subheader("Summary")

def kpi_card(value, label, color="#2f80ed"):
    return f"""
    <div class="kpi-card" style="border-bottom: 4px solid {color};">
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


st.markdown(
    """
    <style>
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 16px;
        padding: 26px 18px;
        text-align: center;
        min-height: 145px;
        box-shadow: 0 4px 14px rgba(15, 42, 58, 0.08);
    }

    .kpi-value {
        font-size: 44px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: 0.5px;
    }

    .kpi-label {
        margin-top: 14px;
        color: #667085;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        kpi_card(f"{len(results_df):,}", "Total Comments", "#2f80ed"),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        kpi_card(round(results_df["clean_comment"].str.len().mean(), 1), "Avg Comment Length", "#f2994a"),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        kpi_card(f"{len(word_counts):,}", "Unique Words", "#27ae60"),
        unsafe_allow_html=True
    )

st.markdown(
    """
    <style>
    .section-title {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .section-title h2 {
        font-size: 34px;
        font-weight: 800;
        color: #1d3557;
        margin: 0;
    }

    .section-accent {
        width: 8px;
        height: 42px;
        border-radius: 8px;
        background: linear-gradient(180deg, #2f80ed, #56ccf2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# Sentiment comparison
# =====================================================

st.markdown(
    """
    <div class="section-title">
        <div class="section-accent"></div>
        <h2>Sentiment Model Comparison</h2>
    </div>
    """,
    unsafe_allow_html=True
)

comparison_table, comparison_long = create_model_comparison(results_df)

st.markdown("#### Model Comparison Table")

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True
)

st.markdown("#### Sentiment Distribution Across 3 Models")

fig = px.bar(
    comparison_long,
    x="sentiment_label",
    y="comment_count",
    color="model",
    barmode="group",
    text="comment_count"
)

fig.update_traces(textposition="outside")

fig.update_layout(
    height=500,
    margin=dict(l=10, r=10, t=30, b=40),
    xaxis_title="Sentiment Label",
    yaxis_title="Comment Count",
    legend_title="Model"
)

st.plotly_chart(fig, use_container_width=True)


# =====================================================
# Individual model charts
# =====================================================


st.markdown(
    """
    <div class="section-title">
        <div class="section-accent"></div>
        <h2>Individual Sentiment Results</h2>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["VADER", "DistilBERT", "RoBERTa"])

with tab1:
    vader_summary = (
        results_df["vader_label"]
        .value_counts()
        .reset_index()
    )
    vader_summary.columns = ["sentiment_label", "comment_count"]

    fig = px.pie(
        vader_summary,
        names="sentiment_label",
        values="comment_count",
        title="VADER Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Top Positive Comments")
    st.dataframe(
        results_df.sort_values("vader_score", ascending=False)[
            ["date", "comment", "vader_label", "vader_score"]
        ].head(10),
        use_container_width=True
    )

    st.write("Top Negative Comments")
    st.dataframe(
        results_df.sort_values("vader_score", ascending=True)[
            ["date", "comment", "vader_label", "vader_score"]
        ].head(10),
        use_container_width=True
    )

with tab2:
    distilbert_summary = (
        results_df["distilbert_label"]
        .value_counts()
        .reset_index()
    )
    distilbert_summary.columns = ["sentiment_label", "comment_count"]

    fig = px.pie(
        distilbert_summary,
        names="sentiment_label",
        values="comment_count",
        title="DistilBERT Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Top Positive Comments")
    st.dataframe(
        results_df[results_df["distilbert_label"] == "positive"]
        .sort_values("distilbert_score", ascending=False)[
            ["date", "comment", "distilbert_label", "distilbert_score"]
        ].head(10),
        use_container_width=True
    )

    st.write("Top Negative Comments")
    st.dataframe(
        results_df[results_df["distilbert_label"] == "negative"]
        .sort_values("distilbert_score", ascending=False)[
            ["date", "comment", "distilbert_label", "distilbert_score"]
        ].head(10),
        use_container_width=True
    )

with tab3:
    roberta_summary = (
        results_df["roberta_label"]
        .value_counts()
        .reset_index()
    )
    roberta_summary.columns = ["sentiment_label", "comment_count"]

    fig = px.pie(
        roberta_summary,
        names="sentiment_label",
        values="comment_count",
        title="RoBERTa Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Top Positive Comments")
    st.dataframe(
        results_df[results_df["roberta_label"] == "positive"]
        .sort_values("roberta_score", ascending=False)[
            ["date", "comment", "roberta_label", "roberta_score"]
        ].head(10),
        use_container_width=True
    )

    st.write("Top Negative Comments")
    st.dataframe(
        results_df[results_df["roberta_label"] == "negative"]
        .sort_values("roberta_score", ascending=False)[
            ["date", "comment", "roberta_label", "roberta_score"]
        ].head(10),
        use_container_width=True
    )

st.divider()


# =====================================================
# Wordcloud and treemap
# =====================================================

st.markdown(
    """
    <div class="section-title">
        <div class="section-accent"></div>
        <h2>Word Frequency Analysis</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("#### Word Cloud")
wc_fig = create_wordcloud(word_counts)
st.pyplot(wc_fig, use_container_width=True)

st.markdown(f"#### Top {top_words} Words Treemap")
tree_fig = create_treemap(word_counts, top_n=top_words)

tree_fig.update_layout(
    height=500,
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(tree_fig, use_container_width=True)

st.divider()


# =====================================================
# Topic modeling section
# =====================================================

st.markdown(
    """
    <div class="section-title">
        <div class="section-accent"></div>
        <h2>Topic Modeling</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "Use the box below to exclude extra words from topic modeling only. "
    "This will not rerun the sentiment models."
)


def save_topic_excluded_words():
    st.session_state["topic_excluded_words"] = st.session_state["topic_excluded_words_input"]


if "topic_excluded_words" not in st.session_state:
    st.session_state["topic_excluded_words"] = ""

st.text_input(
    "Optional words to exclude from topics, separated by commas",
    value=st.session_state["topic_excluded_words"],
    placeholder="Example: people, like, lol",
    key="topic_excluded_words_input"
)

st.button(
    "Rerun Topic Modeling Only",
    on_click=save_topic_excluded_words
)

excluded_words_for_topics = st.session_state.get("topic_excluded_words", "")

lda_tab, ngram_tab, tfidf_tab = st.tabs(
    ["LDA Topics", "N-gram Phrases", "TF-IDF Keywords"]
)

with lda_tab:
    with st.spinner("Running LDA topic modeling..."):
        df_topic, topic_keywords_df, topic_summary = run_lda_topic_modeling(
            df_nlp,
            num_topics=num_topics,
            topn=8,
            user_stopwords_text=excluded_words_for_topics
        )

    st.session_state["latest_df_topic"] = df_topic if not df_topic.empty else None

    if topic_summary.empty:
        st.warning("LDA could not generate topics. Try increasing the number of comments or removing fewer words.")

    else:
        st.markdown("#### Topic Keywords")

        topic_display = topic_keywords_df.copy()
        topic_display["topic_keywords"] = topic_display["topic_keywords"].str.wrap(45)

        st.dataframe(
            topic_display,
            use_container_width=True,
            hide_index=True,
            height=220
        )

        st.markdown("#### Main Discussion Topics")

        fig = px.bar(
            topic_summary,
            x="topic_name",
            y="comment_count",
            text="comment_count"
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=80),
            xaxis_tickangle=-30,
            xaxis_title="Topic",
            yaxis_title="Comment Count"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Comments with Assigned LDA Topics")

        st.dataframe(
            df_topic[
                [
                    "date",
                    "comment",
                    "topic_id",
                    "topic_score"
                ]
            ].sort_values("topic_score", ascending=False).head(50),
            use_container_width=True,
            hide_index=True
        )

with ngram_tab:
    st.markdown("#### Top 3-Word Phrases")

    ngram_df = run_ngram_analysis(
        df_nlp,
        user_stopwords_text=excluded_words_for_topics,
        top_n=20
    )

    if ngram_df.empty:
        st.warning("No 3-word phrases found. Try increasing comment count or lowering filtering rules.")
    else:
        st.dataframe(
            ngram_df,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            ngram_df.sort_values("count"),
            x="count",
            y="three_word_phrase",
            orientation="h",
            text="count",
            title="Top Repeated 3-Word Phrases"
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            height=600,
            margin=dict(l=10, r=10, t=40, b=40),
            xaxis_title="Frequency",
            yaxis_title="3-Word Phrase"
        )

        st.plotly_chart(fig, use_container_width=True)

with tfidf_tab:
    st.markdown("#### Top TF-IDF Keywords and Phrases")

    tfidf_df = run_tfidf_analysis(
        df_nlp,
        user_stopwords_text=excluded_words_for_topics,
        top_n=20
    )

    if tfidf_df.empty:
        st.warning("No TF-IDF terms found. Try increasing comment count or removing fewer words.")
    else:
        st.dataframe(
            tfidf_df,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            tfidf_df.sort_values("tfidf_score"),
            x="tfidf_score",
            y="term",
            orientation="h",
            text="tfidf_score",
            title="Top TF-IDF Keywords and Phrases"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            height=600,
            margin=dict(l=10, r=10, t=40, b=40),
            xaxis_title="Average TF-IDF Score",
            yaxis_title="Keyword / Phrase"
        )

        st.plotly_chart(fig, use_container_width=True)

st.divider()


# =====================================================
# Download results
# =====================================================

st.markdown(
    """
    <div class="section-title">
        <div class="section-accent"></div>
        <h2>Download Results</h2>
    </div>
    """,
    unsafe_allow_html=True
)


csv = results_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download sentiment results as CSV",
    data=csv,
    file_name="youtube_comment_sentiment_results.csv",
    mime="text/csv"
)

latest_df_topic = st.session_state.get("latest_df_topic")

if latest_df_topic is not None and not latest_df_topic.empty:
    topic_csv = latest_df_topic.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download topic results as CSV",
        data=topic_csv,
        file_name="youtube_comment_topic_results.csv",
        mime="text/csv"
    )
