from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import os
import openai
import gradio as gr
from langchain_google_vertexai import VertexAI, VertexAIModelGarden
from new_llms import VertexAIModelGardenPeft

credential_path = "<service_account_key_file>"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# llm = VertexAIModelGarden(endpoint_id='<endpoint_id>', project="<project_id>")
llm = VertexAIModelGardenPeft(endpoint_id='<endpoint_id>', project="<project_id>", location='<region>')
# print(llm.invoke(input="What is the capital of India"))

ans = llm.invoke(input="What is the capital of India",
                 allowed_model_args={"max_tokens": 50,
                                     "temperature": 1.0,
                                     "top_p": 1.0,
                                     "top_k": 10})

print(ans)
# question = "What day comes after Friday?"
# llm(question)


# os.environ["OPENAI_API_KEY"] = "<open ai key>"  # Replace with your key
#
# llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo')

# def predict(message, history):
#     history_langchain_format = []
#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))
#     history_langchain_format.append(HumanMessage(content=message))
#     gpt_response = llm(history_langchain_format)
#     return gpt_response.content

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    print(history_langchain_format)
    gpt_response = llm.invoke(input = message)
    return gpt_response


gr.ChatInterface(predict).launch(server_name="0.0.0.0")
