import streamlit as st
import pickle
import re

# 1. Load the models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# 2. Define the cleaning function (Must match your training logic)
def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# 3. Streamlit UI Layout
def main():
    st.set_page_config(page_title="AI Resume Screener", layout="centered")
    
    st.title("Resume Category Predictor")
    st.markdown("Upload a resume or paste text to see the predicted job category.")

    # Sidebar for instructions
    st.sidebar.title("About")
    st.sidebar.info("This app uses a trained Machine Learning model to categorize resumes into fields like Data Science, Java Developer, etc.")

    # Input Area
    input_text = st.text_area("Paste Resume Content Here:", height=300)

    if st.button("Predict Category"):
        if input_text.strip() != "":
            # Process the input
            cleaned_text = clean_resume(input_text)
            
            # Vectorize the text using the loaded TF-IDF
            vectorized_text = tfidf.transform([cleaned_text])
            
            # Make prediction
            prediction_id = clf.predict(vectorized_text)[0]
            
            # Display Result
            st.success(f"### Predicted Category: **{prediction_id}**")
        else:
            st.warning("Please paste some text first!")

if __name__ == "__main__":
    main()