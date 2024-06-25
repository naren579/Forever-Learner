import streamlit as st
from openai import OpenAI
from scipy import stats
import pandas as pd
import numpy as np
import re

st.set_page_config(layout="wide")
st.title("DOXSPLORE")
with st.sidebar:
    openai_model = st.selectbox('Select the model',['gpt-4o','gpt-4-turbo','gpt-3.5-turbo'])
    API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    dataset= st.file_uploader("Upload your input file", type=["csv"])
if dataset is not None:
    dataset=pd.read_csv(dataset,parse_dates=True)

    for i in dataset.select_dtypes('object').columns:
        print(i)
        if len(dataset[i].mode())>1:
            try:
                dataset[i] = pd.to_numeric(dataset[i])
            except ValueError as e:
                rand_index=np.random.choice(dataset[i][~dataset[i].isna()].index,1)
                #x=str(e)
                x_str=re.sub(r'\d+', '', dataset[i][rand_index].iloc[0])
                if (bool(re.search(r'\d+', dataset[i][rand_index].iloc[0])) == True) and (len(x_str)<(len(dataset[i][rand_index].iloc[0])-len(x_str))):
                    dataset[i] = pd.to_numeric(dataset[i].str.replace(',',''),errors='coerce')
                    #print(f'column "{i}" has been converted to Numeric')
                    continue
                else:
                    continue
        else:
            continue

# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# import pprint
# import os


###Enivironment settings for openai API key and Vector Embeddings############

client = OpenAI(api_key=API_KEY)   
persist_directory = '/mount/src/emplochat/embeddings/db'

#################Initializing class #########################################
class Analyze_csv():
    def __init__(self,df):
        self.df = df

    def no_of_rows(self):
        return len(self.df)
        
    def no_of_null_total(self):
        x=''
        for i,j in zip(self.df.columns,self.df.isna().sum()):
            x = x+f'"{i}" column has {j} null values'+'\n'
        return x
    def col_details(self):
        total_cols=len(self.df.columns)
        num_cols='numerical columns: \n'
        cat_cols='Categorical columns: \n'

        for i in self.df.select_dtypes('number').columns:
            if len(self.df[i].value_counts())>25:
                num_cols = num_cols+i+'\n'
            else:
                cat_cols = cat_cols+i+'\n'

        for j in self.df.select_dtypes('object').columns:
            cat_cols = cat_cols+j+'\n'
        return total_cols, num_cols, cat_cols

    def num_col_details(self):
        a='Mean for Numerical columns:\n'
        b='Min for Numerical columns:\n'
        c='Max for Numerical columns:\n'
        d='Median for Numerical columns:\n'
        e='standard deviation for Numerical columns:\n'
        
        for i in self.df.select_dtypes('number').columns:
            a=a+f'{i} : {str(round(self.df[i].mean(),2))}'+'\n'
            b=b+f'{i} : {str(round(self.df[i].min(),2))}'+'\n'
            c=c+f'{i} : {str(round(self.df[i].max(),2))}'+'\n'
            d=d+f'{i} : {str(round(self.df[i].median(),2))}'+'\n'
            e=e+f'{i} : {str(round(self.df[i].std(),2))}'+'\n'
            
        return a,b,c,d,e
    def cat_col_details(self):
        a='Mode for the Categorical columns:\n'
        b='Top 5 values for Categorical columns:\n'
        
        for i in self.df.select_dtypes('object').columns:
            if len(self.df[i].mode()) > 1:
                a=a+f'{i} :contains more than 1 mode'+'\n'
            else:
                a=a+f'{i} : {str(self.df[i].mode()[0])}'+'\n'
        for i in self.df.select_dtypes('object').columns:
            if len(self.df[i].mode()) == 1:
                b=b+f'{i} : {dict(self.df[i].value_counts()[0:5])}'+'\n'
            else:
                b=b+f'{i} : contains more than 1 mode'+'\n'
                 
        return a,b
        
    def outliers(self):
        Q1 = round(self.df.select_dtypes('number').quantile(0.25),2)
        Q3 = round(self.df.select_dtypes('number').quantile(0.75),2)
        IQR = Q3 - Q1
        x='Outliers for Numerical columns:\n'

        for i in self.df.select_dtypes('number').columns:

            x=x+f'{i} : {str(len(self.df[~((self.df[i]>(Q1[i]-1.5*IQR[i])) & (self.df[i]<(Q3[i]+1.5*IQR[i])))][i])-self.df[i].isna().sum())}'+'\n'
            
        return x
    
    def distribution_type(self):
        a='Skewness for Numerical columns:\n'
        b='Kurtosis for Numerical columns:\n'
        c='Shapiro-wilk test for Numerical columns:\n'
        for i in self.df.select_dtypes('number').columns:
            a=a+f'{i} : {self.df[~self.df[i].isna()][i].skew()}\n'
            b=b+f'{i} : {self.df[~self.df[i].isna()][i].kurtosis()}\n'
            
            shapiro_test = stats.shapiro(self.df[~self.df[i].isna()][i])
            if shapiro_test[1] > 0.05:
                c = c+f'{i} : This column appear to be normally distributed\n'
            else:
                c = c+f'{i} : This column does not appear to be normally distributed\n'

        return a,b,c

    def correlation(self):
        a = 'The correlation coefficient between the numerical columns are given below:\n'
        for i in self.df.select_dtypes('number').columns:
            for j in self.df.select_dtypes('number').columns:
                corr=self.df[[i,j]].corr().iloc[0,1]
                if i!=j and (corr>0.75 or corr<-0.75):
                    a=a+f'{i} <-> {j} : {corr}\n'
                else:
                    continue
        return a     
##############################################################################


#################Initialize session state to store history####################



if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = openai_model

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
      
    

# Accept user input

if query := st.chat_input("Enter your query here?"):
    a1=Analyze_csv(dataset)
    context=f'''
         [INST]
         Below are the details of a dataset. 

         Role: your task is to answer the queries on the dataset by 
         referring the below details, answers for the questions has to be
         very accurate and to the point. 
         
         If user asks about the simple summary of the dataset, 
         by referring dependant and independant columns, 
         generate a short summary of the dataset without including
         in depth numerical details(dont include median,max,min,mean etc) 
         and explain what might be the dataset is about theoritically.

         Note: dont give unneccesary details like mean, median etc to user unless you 
         are specifically asked about that.
         --------------------------------------------------------------
         Details:
         Total number of records {a1.no_of_rows()}
         
         Null values \n{a1.no_of_null_total()}
         
         Total columns : {a1.col_details()[0]}
         
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

    prompt = f''' 
                {context}
                Question: {query}
                [/INST]
                '''        
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
          

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(max_tokens=800,
                model=st.session_state["openai_model"],
                messages=[{"role": "system", "content":prompt}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    

    

# ###############################################################################
















