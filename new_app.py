import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import fsspec 

# Load the preprocessed data
df = pd.read_csv('final_data.csv')
#df = pd.read_csv('final_data.csv',encoding='ISO-8859-1')

# Define the TD-IDF vectorizer and fit it to the data
tdidf = TfidfVectorizer()
tdidf.fit(df['Question'].str.lower())

# Define the support vector machine model and fit it to the data
svc_model = SVC(kernel='linear')
svc_model.fit(tdidf.transform(df['Question'].str.lower()), df['Class'])

# Define a function to get the answer to a given question
def get_answer(question):
    # Vectorize the question
    question_tdidf = tdidf.transform([question.lower()])
    
    # Calculate the cosine similarity between both vectors
    cosine_sims = cosine_similarity(question_tdidf, tdidf.transform(df['Question'].str.lower()))

    # Get the index of the most similar text to the query
    most_similar_idx = np.argmax(cosine_sims)

    # Get the predicted class of the query
    predicted_class = svc_model.predict(question_tdidf)[0]
    
    # If the predicted class is not the same as the actual class, return an error message
    if predicted_class != df.iloc[most_similar_idx]['Class']:
        return 'Sorry could not find an appropriate answer. Kindly contact the bank via customer care number'
    
    # Get the answer and construct the response
    answer = df.iloc[most_similar_idx]['Answer']
    response = f"Answer: {answer}"
    
    return response

# Create a streamlit app
def app():
    # Set the app title
    st.set_page_config(page_title="Banking Chatbot", layout="wide")

    # Add a title and description to the app
    st.subheader("Welcome to Banking Chatbot")
    st.title("Hi!My name is Emily and I'm a ChatBot")
    st.write("I can help you with any bank related queries. Please type your question in the space provided.")

    # Create a text input for the user to ask a question
    question = st.text_input("Type your question")

    # Add a button to submit the question
    if st.button("Submit"):
        # Check if the user has entered a question
        if question == "":
            st.warning("Please enter a question.")
        else:
            # Call the get_answer function to predict the answer to the question
            answer = get_answer(question)

            # Display the answer to the user
            st.success(answer)

# Run the streamlit app
if __name__ == '__main__':
    app()
