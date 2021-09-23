# %%
import streamlit as st
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tokenizers import AddedToken
import time

@st.cache(hash_funcs={"tokenizers.AddedToken": lambda _: 1,
                      "_regex.Pattern": lambda _: 1},
                      allow_output_mutation=True) # 👈 This function will be cached
def iniliaze_model():
    # Do something really slow in here!
    tokenizer = RobertaTokenizer.from_pretrained('nielsr/codet5-small-code-summarization-ruby')
    model = T5ForConditionalGeneration.from_pretrained('nielsr/codet5-small-code-summarization-ruby')
    return tokenizer,model

agree = st.checkbox('Start')

if agree:
    tokenizer,model = iniliaze_model()
    st.title('CodeT5 Summarize')


    code = st.text_area('Code to analyze', '''
    public class IntegerSumExample1 {  
        public static void main(String[] args) {          
        int a = 65;  
            int b = 35;  
            // It will return the sum of a and b.  
            System.out.println("The sum of a and b is = " + Integer.sum(a, b));  
        }  
    }  
    ''', height=400)


    input_ids = tokenizer(code, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)

    st.subheader('Summary')
    st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
