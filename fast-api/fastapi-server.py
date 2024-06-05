from typing import Optional, List, Dict
from fastapi import FastAPI, responses
from configparser import ConfigParser
import torch
import logging
import importlib
import os

from huggingface_hub import login
login(token="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")

global device
global processor
global models
global tokenizers
global logger
global default_question, default_context


import torch_neuronx
import torch
from transformers import AutoTokenizer
from transformers_neuronx import constants
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.mistral.model import MistralForSampling


logger = logging.getLogger()

# Read static configuration from config.properties
logger.warning("\nParsing configuration ...")

with open('config.properties') as f:
    config_lines = '[global]\n' + f.read()
    f.close()
config = ConfigParser()
config.read_string(config_lines)

num_models_per_server = int(config['global']['num_models_per_server'])

env_var = os.getenv('NEURON_RT_VISIBLE_CORES')
neuron_core = int(env_var[0])
print(neuron_core)

# FastAPI server
app = FastAPI()


# Server healthcheck
@app.get("/")
async def read_root():
    return {"Status": "Healthy"}


prediction_api_name = f'predictions_neuron_core_{neuron_core}'
postprocess = True
quiet = False


global tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


global model
model = MistralForSampling.from_pretrained(
           "mistralai/Mistral-7B-Instruct-v0.1",
            batch_size=1,
            tp_degree=2,
            n_positions=8000,
            amp="bf16",
            context_length_estimate=8000
            )
            # Load the compiled Neuron artifacts
model.load("model.pt")
model.to_neuron()


# Model inference API endpoint
@app.post("/{prediction_api_name}/{model_id}")
async def infer(model_id, prompt):
    print(model_id)
    # print(messages)
    status = 200
    messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Do you have any recipes for the mayo you're talking about?"},
            ]
    
    
    encodeds = tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            )
    del messages
    
    # Use torch.inference_mode() as a context manager to optimize memory usage
    with torch.inference_mode():
        # Perform inference
        with torch.no_grad():
            output = model.sample(encodeds, sequence_length=6000, start_ids=None)
            print([tokenizer.decode(tok) for tok in output])
            del encodeds
            del output

    # Manually delete tensors to free up memory
    
    

    return responses.JSONResponse(status_code=status, content={"detail": ""})

