import torch
from transformers import T5Tokenizer,T5ForConditionalGeneration,AutoTokenizer
import numpy as np
import os
import streamlit as st
st.set_page_config(layout="wide")

def summarize_text(text, model, tokenizer, max_length=512, num_beams=5):
    # Preprocess the text
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True
    )

    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=512,
        num_beams=num_beams,
        # early_stopping=True,
    )

    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

from huggingface_hub import HfApi

with st.sidebar:
  os.environ['HUGGINGFACE_TOKEN'] = st.text_input('Enter your huggingface api key here')
model_name = 'Naren579/T5-Text-summarize'

model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

text=st.text_input('Enter here:')

if st.button('Summarize'):
   summary = summarize_text(text, model, tokenizer)
   st.write(summary)