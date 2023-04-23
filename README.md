# Banking-Chatbot
Developed a banking chatbot which can guide a person with any banking queries.

# Description of files
BankFAQ.csv files contains the original Bank's FAQ dataset.
final_data.csv contains original Bank's FAQ dataset and New Bank FAQ dataset. This dataset has been used for model prediction

Banking Chatbot project with Class(DV).ipynb file contains trained data, EDA, Feature Engineering, preprocessing data using NLP, Model prediction and accuracy.

The data has been preprocessed by using stemming and tf-idf vectorizing the questions The same process is applied to user's query.
Got highest accuracy in Support Vector Machine with linear kernel. Once the class is found, I defined a subset of questions belonging to this class and then used Cosine Similarity to find the most likely question. The answer associated with the question with maximum cosine similarity to user's query is served to the user.

new_app.py has streamlit codes which is used for deployment in Streamlit.
