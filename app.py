import streamlit as st
import pickle
import os
import pandas as pd
import plotly.express as px

MODEL_PATH = "model/sentiment_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Please run `python train.py` first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def get_emoji(sentiment):
    return {"positive": "😊", "neutral": "😐", "negative": "😠"}.get(sentiment, "❓")

def get_color(sentiment):
    return {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}.get(sentiment, "#bdc3c7")

# --- Page Config ---
st.set_page_config(
    page_title="Airline Tweet Sentiment",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Airline Tweet Sentiment Analyzer")
st.markdown("Analyze the sentiment of airline-related tweets using a Machine Learning model.")
st.divider()

model = load_model()

# --- Single Tweet ---
st.subheader("🔍 Analyze a Tweet")
tweet_input = st.text_area(
    "Enter a tweet:",
    placeholder="e.g. @AmericanAir your service was absolutely terrible today!",
    height=100
)

if st.button("Predict Sentiment", type="primary"):
    if tweet_input.strip():
        prediction = model.predict([tweet_input])[0]
        proba = model.predict_proba([tweet_input])[0]
        classes = model.classes_

        emoji = get_emoji(prediction)
        color = get_color(prediction)

        st.markdown(f"""
        <div style='background-color:{color}22; border-left: 5px solid {color};
                    padding: 16px; border-radius: 8px; margin-top: 12px'>
            <h3 style='color:{color}; margin:0'>{emoji} {prediction.capitalize()}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Confidence Scores:**")
        prob_df = pd.DataFrame({"Sentiment": classes, "Confidence": proba})
        fig = px.bar(prob_df, x="Sentiment", y="Confidence",
                     color="Sentiment",
                     color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
                     range_y=[0, 1])
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter a tweet first.")

st.divider()

# --- Batch Analysis ---
st.subheader("📋 Batch Analyze Multiple Tweets")
batch_input = st.text_area(
    "Enter multiple tweets (one per line):",
    placeholder="Tweet 1\nTweet 2\nTweet 3",
    height=150
)

if st.button("Analyze All"):
    tweets = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
    if tweets:
        predictions = model.predict(tweets)
        results = pd.DataFrame({"Tweet": tweets, "Sentiment": predictions})
        results["Emoji"] = results["Sentiment"].apply(get_emoji)

        st.dataframe(results[["Tweet", "Emoji", "Sentiment"]], use_container_width=True)

        summary = results["Sentiment"].value_counts().reset_index()
        summary.columns = ["Sentiment", "Count"]
        fig2 = px.pie(summary, names="Sentiment", values="Count",
                      color="Sentiment",
                      color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Please enter at least one tweet.")

st.divider()
st.caption("Built with Scikit-learn + Streamlit | Airline Twitter Sentiment Dataset")
