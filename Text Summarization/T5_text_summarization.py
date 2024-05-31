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
	
output = query({
	"inputs": st.text_input('ENter your text here:'),
})

# st.write(output[0][0]['label'])
if st.button('Summarize') and (API_TOKEN is not None):
	st.write(output)

# 	if (output[0][0]['label']) == 'LABEL_0':
# 		st.markdown("# The Sentence Seems to be POSITIVE")
# 		st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1C4VPejYDvywKmk12MHyeH1z0ubr0E1A8lg&usqp=CAU')
# 	elif (output[0][0]['label']) == 'LABEL_1':
# 		st.markdown("# The Sentence Seems to be NEGATIVE")
# 		st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSbbuDRvaFBgko-Kox-TUykBQFIqGU7p5SWt5kFoKK1p9B_LQWlPbswDfiJH6RpEGfqQbY&usqp=CAU')
# 	else:
# 		st.markdown("# The Sentence Seems to be NEUTRAL")
# 		st.image('https://assets-global.website-files.com/5bd07788d8a198cafc2d158a/61c49a62dccfe690ca3704be_Screen-Shot-2021-12-23-at-10.44.27-AM.jpg')
# else:
# 	st.error('Please provide your HuggingFace API key to continue.')


