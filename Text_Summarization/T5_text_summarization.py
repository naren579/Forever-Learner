import requests
import streamlit as st
with st.sidebar:
	API_TOKEN=st.text_input('Enter your huggingface api key here')
	if st.button('Submit'):
		if API_TOKEN is not None:
			st.success('Token submitted successfully!')
API_URL = "https://api-inference.huggingface.co/models/Naren579/T5-Text-summarize"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def summarize_text(text):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 512,  # Adjust as needed
            "min_length": 50,   # Adjust as needed
            "length_penalty": 2.0,
            "num_beams": 4,
        }
    }
    response = query(payload)
#[0]['summary_text']
    return response  # Adjust based on actual response structure
    

# with st.sidebar:
#     API_TOKEN = st.text_input('Enter your huggingface API key here')
#     if st.button('Submit'):
#         if API_TOKEN:
#             st.success('Token submitted successfully!')

text_input = st.text_input('Enter your text here:')
text=text_input
if st.button('Summarize') and API_TOKEN:
    summarized_text = summarize_text(text)
    st.write(summarized_text[0]['generated_text'])
