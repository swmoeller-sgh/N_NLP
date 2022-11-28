"""



# https://huggingface.co/course/chapter1/1?fw=pt

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I don't like Mondays"))


from transformers import pipeline

list_labels = ["teaching", "writing", "sports"]

classifier = pipeline("zero-shot-classification")

for item in list_labels:
    result = classifier(
        "This is a course about the Transformers library. It will teach, how to play badminton.",
        candidate_labels=item)
    print(result)


from transformers import pipeline

generator = pipeline("text-generation", "gpt2")
print(generator("Tonight, my wife and I will",
                max_length=30,
                num_return_sequences=2
               ))


from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2Model.from_pretrained('gpt2')
text = "Tonight, my wife and I will"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)



from transformers import pipeline

unmasker = pipeline("fill-mask")
print(unmasker("This course will teach you all about <mask> models.", top_k=2))


# Named entity recognition
# Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to
# entities such as persons, locations, or organizations. Let’s look at an example:

from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
# to tell the pipeline to regroup together the parts of the sentence that correspond to the same entity:
print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

from transformers import pipeline

question_answerer = pipeline("question-answering")
print(question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
))

from transformers import pipeline

summarizer = pipeline("summarization")
result= summarizer(
    "
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"
)
print(result)

from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result=translator("Ce cours est produit par Hugging Face.")
print(result)

from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])


from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(tokenizer)
print(type(tokenizer))
print(inputs.input_ids)
print(inputs.attention_mask)


from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)



from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "Finally there is light at the end of the tunnel!",
    "The light at the end of the tunnel was a train taking me to hell!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
for key, value in inputs.items():
    print (key,"\n", value)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs)
print(type(outputs))
print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs)
print(outputs.logits.shape)

print(outputs.logits)
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)

from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

# Building the config
config = BertConfig()

# Building the model from the config
model1 = BertModel(config)
model2 = BertModel.from_pretrained("bert-base-cased")
print("model 1: \n", model1.config)
print("model 2: \n", model2.config)

model2.save_pretrained("/Users/swmoeller/delete")

sequences = ["Hello!", "Cool.", "Nice!"]
print(sequences)
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(inputs)

"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result=tokenizer("Using a Transformer network is simple")

print(result)

tokenizer.save_pretrained("/Users/swmoeller/delete")
