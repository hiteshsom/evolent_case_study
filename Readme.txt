The notebook file (.ipynb) is used for model fine-tuning and deployment to an endpoint.
You can go to Model Garden choose Falcon Instruct 7B model and open a google colab notebook from there.
It will be easier to execute fine tuning from there instead of running this notebook on local as there will less
authentication related issues in colab. The only change you have to make in colab is you have to copy paste code
for preprocess_data function that I have written in the notebook file. You can find that code in the fintune section.

That code I am also pasting here:
from google.cloud import storage

# BUCKET_NAME = "gs://my_bucket"

storage_client = storage.Client()
bucket = storage_client.get_bucket('<cloud storage bucket name>')

# blob = bucket.blob('train.dat')
# blob = blob.download_as_string()
# blob = blob.decode('utf-8')

import json
# {
#     "description": "Template used by Alpaca-LoRA.",
#     "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
#     "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
#     "response_split": "### Response:"
# }

def preprocess_data(bucket, dataset):
  blob = bucket.blob(f"{dataset}.dat")
  blob = blob.download_as_string()
  blob = blob.decode('utf-8')
  data_examples = blob.split('\n')
  tab_char = '\t'
  print(type(data_examples))
  label_map = {'1': 'digestive system diseases',
               '2': 'cardiovascular diseases',
               '3': 'neoplasms',
               '4': 'nervous system diseases',
               '5': 'general pathological conditions'}
  final_data = []
  try:
    for example in data_examples:
      final_data.append(str({'input_text': f"You are a medical conversational bot. Classify the given text into an appropriate disease group. The groups are: [digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, general pathological conditions] Text: {example.split(tab_char)[1]}", 'output_text': f"{label_map[example.split(tab_char)[0]]}"}))
  except Exception as e:
    pass
  print(len(final_data))
  print(final_data[0])

  # instruction = 'Describe the type of disease'
  # ex_desc = {
  #               "description": "Template used by Alpaca-LoRA.",
  #               "prompt_input": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
  #               "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
  #               "response_split": "### Response:"
  #           }
  data_jsonl = "\n".join(final_data)

  # with open(f'{dataset}.jsonl', 'w') as f:
  #   f.write(data_jsonl)

  # create a blob
  blob = bucket.blob(f'{dataset}.jsonl')
  # upload the blob
  blob.upload_from_string(
        data=json.dumps(data_jsonl)
        )

  # return data_examples


Also to run the finetuning you need to create a service account, a cloud bucket storage where train.data and test.dat resides.
Make sure your cloud storage bucket and notebook are in same region.

I was not able to fine tune Falcon because even after trying multiple times, GCP did not have GPU available for CustomTraining.
However they had GPU available for Serving or inferencing. Hence I could deploy the a pretrained model using the colab notebook.
The deployment creates an endpoint, you have to write the endpoint id in chat.py (arguments to this constructor VertexAIModelGardenPeft)
which will enable the langchain to use the deployed model.

You must keep the service account key file (Its a json file) in the docker only. Or provide a correct path to it in
credential_path variable in chat.py. You will have to download service account key file from GCP.

The Dockerfile is there which will install all the packages to run chat interface and langchain.


Reason for choosing Falcon as llm
1. The case study said to use the train data and finetune a model for a conversational bot.
BERT is usually good for finetuning for classification task but it may not be a good option for
conversational bot. Hence I had to choose a text generation model. I chose Falcon Instruct 7B since
it can also be finetuned for classification and its also suitable for a conversational bot.

Challenges:
1. Hallucinations are a general problem in LLMs, so we need to see the accuracy of classification after finetuning Falcon

Other improvements:
1. In this case we havent used any agents in Langchain nor we have set a Langchain chain such that it remembers past
conversation and include it in prompt for historical context. Since most of the time was spent on trying to finetune model in GCP I could not use
these langchain features. But overall the code provides a proof of concept of how the whole system would work.

