"""

# Try it out! Apply the tokenization manually on the two sentences used in section 2 (“I’ve been waiting for a
# HuggingFace course my whole life.” and “I hate this so much!”). Pass them through the model and check that you get
# the same logits as in section 2. Now batch them together using the padding token, then create the proper attention
# mask. Check that you obtain the same results when going through the model!

# sequence01 = "I’ve been waiting for a HuggingFace course my whole life."
# Logits: tensor([[-2.5720,  2.6852]], grad_fn=<AddmmBackward0>)
# sequence02 = "I hate this so much!"
# Logits: tensor([[ 3.1931, -2.6685]], grad_fn=<AddmmBackward0>)

sequence01 = ["I’ve been waiting for a HuggingFace course my whole life.", "I hate this so much!"]

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# tokens = tokenizer.tokenize(sequence01)
tokens = tokenizer(sequence01, padding=True, truncation=True, return_tensors="pt")

ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
"""


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)