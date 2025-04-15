import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# Download stopwords once
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('/mount/src/tweet_analyzer/model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('/mount/src/tweet_analyzer/vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load model files: {str(e)}")
        return None, None  # Graceful fallback
# Initialize Nitter scraper with better instance management
@st.cache_resource
def initialize_scraper():
    try:
        # Current working instances (check https://github.com/zedeus/nitter/wiki/Instances)
        working_instances = [
            "https://nitter.net",
            "https://nitter.it",
            "https://nitter.domain.glass",
            "https://nitter.poast.org"
        ]
        
        # Try each instance with timeout
        for instance in working_instances:
            try:
                scraper = Nitter(
                    log_level=1,
                    skip_instance_check=False,
                    instance=instance
                )
                # Quick test with minimal request
                test = scraper.get_tweets("twitter", mode='user', number=1, max_retries=1)
                if test and 'tweets' in test:
                    st.success(f"Connected to Nitter instance: {instance}")
                    return scraper
            except Exception as e:
                continue
        
        # If all instances fail, try with skip_check as last resort
        scraper = Nitter(
            log_level=1,
            skip_instance_check=True
        )
        st.warning("Using fallback mode with instance check disabled")
        return scraper
        
    except Exception as e:
        st.error(f"Scraper initialization failed: {str(e)}")
        return None

def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return "Negative" if model.predict(vectorizer.transform([text]))[0] == 0 else "Positive"

def create_card(tweet_text, sentiment):
    color = "#4CAF50" if sentiment == "Positive" else "#F44336"
    return f"""
    <div style="background-color: {color}; padding: 12px; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
        <h5 style="color: white; margin-top: 0;">{sentiment} Sentiment</h5>
        <p style="color: white; margin-bottom: 0;">{tweet_text}</p>
    </div>
    """

def main():
    import os
    st.write("Files in directory:", os.listdir('/mount/src/tweet_analyzer'))
    
    # Rest of your code...
    st.title("Twitter Sentiment Analysis 🐦")
    st.markdown("Analyze text or recent tweets from public accounts")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    tab1, tab2 = st.tabs(["🔤 Analyze Text", "🐦 Analyze Tweets"])
    
    with tab1:
        text_input = st.text_area("Enter text to analyze:", placeholder="Type your text here...")
        if st.button("Analyze Sentiment", type="primary"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.markdown(create_card(text_input, sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze")

    with tab2:
        if not scraper:
            st.error("""
            Twitter scraping service is currently unavailable. 
            Possible reasons:
            - All Nitter instances are down
            - Twitter has blocked Nitter access
            - Network restrictions
            
            Please try:
            1. Using text analysis instead
            2. Checking Nitter status
            3. Coming back later
            """)
            st.link_button("Check Nitter Status", "https://github.com/zedeus/nitter/wiki/Instances")
        else:
            username = st.text_input("Enter Twitter username:", 
                                   placeholder="elonmusk", 
                                   help="Without @ symbol. Note: Private accounts won't work")
            
            if st.button("Fetch & Analyze Tweets", type="primary"):
                if not username.strip():
                    st.warning("Please enter a username")
                else:
                    try:
                        with st.spinner(f"Fetching tweets from @{username}..."):
                            tweets_data = scraper.get_tweets(
                                username, 
                                mode='user', 
                                number=5,
                                max_retries=3
                            )
                        
                        if tweets_data and 'tweets' in tweets_data:
                            if tweets_data['tweets']:
                                st.success(f"Found {len(tweets_data['tweets'])} recent tweets")
                                for tweet in tweets_data['tweets']:
                                    tweet_text = tweet['text']
                                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                                    st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
                            else:
                                st.warning("No public tweets found. Account may be private or have no tweets.")
                        else:
                            st.error("Received unexpected response from Nitter")
                            
                    except Exception as e:
                        st.error(f"Error fetching tweets: {str(e)}")
                        st.info("""
                        Troubleshooting tips:
                        1. Try a different username
                        2. Wait a few minutes and retry
                        3. The account might be suspended
                        4. Nitter instances may be overloaded
                        """)

if __name__ == "__main__":
    main()
