import streamlit as st
from openai import OpenAI
from scipy import stats
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("DOXSPLORE")
with st.sidebar:
    API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    dataset = st.file_uploader("Upload your input file", type=["csv"])
if dataset is not None:
    dataset = pd.read_csv(dataset)

### Initialize OpenAI client ###
client = OpenAI(api_key=API_KEY)

################# Initializing class #################################
class Analyze_csv():
    def __init__(self, df):
        self.df = df

    def no_of_rows(self):
        return len(self.df)
        
    def no_of_null_total(self):
        x = ''
        for i, j in zip(self.df.columns, self.df.isna().sum()):
            x = x + f'"{i}" column has {j} null values' + '\n'
        return x

    def col_details(self):
        total_cols = len(self.df.columns)
        num_cols = 'The numerical columns in this dataset are: \n'
        cat_cols = 'The Categorical columns in this dataset are: \n'

        for i in self.df.select_dtypes('number').columns:
            num_cols = num_cols + i + '\n'
        for j in self.df.select_dtypes('object').columns:
            cat_cols = cat_cols + j + '\n'
        return total_cols, num_cols, cat_cols

    def num_col_details(self):
        a = 'Mean(average) value for the Numerical columns are as below:\n'
        b = 'Min value for the Numerical columns are as below:\n'
        c = 'Max value for the Numerical columns are as below:\n'
        d = 'Median value for the numerical columns are as below:\n'
        e = 'Standard deviation of the numerical columns as below:\n'
        
        for i in self.df.select_dtypes('number').columns:
            a = a + f'{i} : {str(round(self.df[i].mean(), 2))}' + '\n'
            b = b + f'{i} : {str(round(self.df[i].min(), 2))}' + '\n'
            c = c + f'{i} : {str(round(self.df[i].max(), 2))}' + '\n'
            d = d + f'{i} : {str(round(self.df[i].median(), 2))}' + '\n'
            e = e + f'{i} : {str(round(self.df[i].std(), 2))}' + '\n'
            
        return a, b, c, d, e

    def cat_col_details(self):
        a = 'Mode for the Categorical columns are as below:\n'
        b = 'Top 5 repeated values for Categorical columns are as below:\n'
        
        for i in self.df.select_dtypes('object').columns:
            if len(self.df[i].mode()) > 1:
                a = a + f'{i} : this column has more than 1 mode' + '\n'
            else:
                a = a + f'{i} : {str(self.df[i].mode()[0])}' + '\n'
        for i in self.df.select_dtypes('object').columns:
            if len(self.df[i].mode()) == 1:
                b = b + f'{i} : {dict(self.df[i].value_counts()[0:5])}' + '\n'
            else:
                b = b + f'{i} : This column has more than 1 mode' + '\n'
                 
        return a, b
        
    def outliers(self):
        Q1 = self.df.select_dtypes('number').quantile(0.25)
        Q3 = self.df.select_dtypes('number').quantile(0.75)
        IQR = Q3 - Q1
        x = 'Outliers for each Numerical column are as below:\n'

        for i in self.df.select_dtypes('number').columns:
            x = x + f'{i} : {str(len(self.df[~((self.df[i]>(Q1[i]-1.5*IQR[i])) & (self.df[i]<(Q3[i]+1.5*IQR[i])))][i])-self.df[i].isna().sum())}' + '\n'
            
        return x
    
    def distribution_type(self):
        a = 'Skewness for each Numerical column are as below:\n'
        b = 'Kurtosis for each Numerical column are as below:\n'
        c = 'Shapiro-wilk test for each Numerical column are as below:\n'
        for i in self.df.select_dtypes('number').columns:
            a = a + f'{i} : {self.df[~self.df[i].isna()][i].skew()}\n'
            b = b + f'{i} : {self.df[~self.df[i].isna()][i].kurtosis()}\n'
            
            shapiro_test = stats.shapiro(self.df[~self.df[i].isna()][i])
            if shapiro_test[1] > 0.05:
                c = c + f'{i} : This column appear to be normally distributed\n'
            else:
                c = c + f'{i} : This column does not appear to be normally distributed\n'

        return a, b, c

    def correlation(self):
        a = 'The correlation coefficient between the numerical columns are given below:\n'
        for i in self.df.select_dtypes('number').columns:
            for j in self.df.select_dtypes('number').columns:
                corr = self.df[[i, j]].corr().iloc[0, 1]
                if i != j and (corr > 0.75 or corr < -0.75):
                    a = a + f'{i} <-> {j} : {corr}\n'
                else:
                    continue
        return a

################# Initialize session state to store history ####################

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to manage conversation history and trim it to stay within token limit
def manage_conversation_history(history, max_tokens=16385, max_new_tokens=1500):
    current_tokens = sum([len(m['content'].split()) for m in history])
    while current_tokens + max_new_tokens > max_tokens:
        history.pop(0)
        current_tokens = sum([len(m['content'].split()) for m in history])
    return history

# Analyze the dataset and prepare the details
if dataset is not None:
    a1 = Analyze_csv(dataset)

    dataset_details = f'''
        Total number of records in the dataset are {a1.no_of_rows()}
        
        Null values for each column are given below\n{a1.no_of_null_total()}
        
        Total columns in the dataset are : {a1.col_details()[0]}
        
        {a1.col_details()[1]}
        
        {a1.col_details()[2]}
        
        {a1.num_col_details()[0]}
        
        {a1.num_col_details()[1]}
        
        {a1.num_col_details()[2]}
        
        {a1.num_col_details()[3]}
        
        {a1.num_col_details()[4]}
        
        {a1.cat_col_details()[0]}
        
        {a1.cat_col_details()[1]}
        
        {a1.outliers()}
        
        {a1.distribution_type()[0]}
        
        {a1.distribution_type()[1]}
        
        {a1.distribution_type()[2]}
        
        {a1.correlation()}
    '''

# Accept user input
if query := st.chat_input("Enter your query here?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Ensure the conversation history stays within the token limit
    st.session_state.messages = manage_conversation_history(st.session_state.messages)

    # Prepare the prompt with dataset details and user query
    prompt = f'''
         [INST]
         Below are the details of a dataset. it covers details like null value count, 
         total rows, data type of the columns,type of distribution, type of relation 
         among the columns. your task is to answer the queries on the dataset by 
         referring the below details, answers the questions accurately. If user asks
         about the range of certain column, refer to maximum and minimum values of the 
         column and answer it.each detail is seperated by an empty line.
         --------------------------------------------------------------
         Details:
         {dataset_details}
         --------------------------------------------------------------
         Question: {query}
         [/INST]
         '''

    # Add assistant system message with dataset details
    st.session_state.messages.append({"role": "system", "content": dataset_details})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Get assistant response
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=st.session_state.messages,
        max_tokens=1500
    )

    response_text = st.write_stream(response)
  
#[1]['choices'][0].message.content
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # # Display assistant response in chat message container
    # with st.chat_message("assistant"):
    #     st.markdown(response_text[1])

# Ensure to keep the input box focused
#st.text_input("You: ", key="input", value='', placeholder="Type your message here...", label_visibility='collapsed')
