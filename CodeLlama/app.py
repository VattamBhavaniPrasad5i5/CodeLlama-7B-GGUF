import gradio as gr
import time
from ctransformers import AutoModelForCausalLM


def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        "codellama-7b-instruct.Q2_K.gguf",
        model_type="llama",
        max_new_tokens=1096,
        repetition_penalty=1.13,
        temperature=0.1,
    )
    return llm


# gradioapp
def llm_funtion(message, chat_history):
    llm = load_llm()
    response = llm(message)
    output_texts = response
    return output_texts


title = "CodeLlama 7B GGUF"

example = [
    "write a code to connect a SQL database and list down all the table",
    "write a python code for linear regression model using scikit learn",
    "write the code to implement a binary tree implementation in c language",
    "what are the benifits of python programming language?",
    "Create a neural network for vgg16",
]
gr.ChatInterface(fn=llm_funtion, title=title, examples=example).launch()
