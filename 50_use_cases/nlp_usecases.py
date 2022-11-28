import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from user_interaction import selectFromDict
import pandas as pd

# variable definition
detailed_log = False

# [USER OPTION] = PROGRAM RESULT
options = {}
options['Text classification: positive - negative sentence'] = 'posneg'
options['Named entity recognition'] = 'ner'
options['Question answering'] = 'quan'
options['Summarization'] = 'summ'
options['Translation DE - RO'] = 'transdero'
options['Text generation'] = 'textgen'
options['close'] = 'close'


def sentence_sentiment_classification(in_detailed_log=False):
    user_prompt = []
    raw_inputs = []
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print("Wichtiges Tool, um im fraenkisch / schwaebischen Sprachraum positiv und negativ gemeinte Saetze zu "
      "klassifizieren!\n")
    """
    while user_prompt != "done":
        user_prompt = input("Enter sentence to be checked (and <done> for ending the input): ")
        if user_prompt != "done":
            raw_inputs.append(user_prompt)
    """
    raw_inputs= ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

    # This architecture contains only the base Transformer module: given some inputs, it outputs what we’ll call hidden
    # states, also known as features. While these hidden states can be useful on their own,
    # they’re usually inputs to another part of the model, known as the head.
    # model = AutoModel.from_pretrained(checkpoint)
    # print("Output shape (batch size, sequence length, hidden size (i.e. the vector dimension of each model input): \n",
    #       outputs.last_hidden_state.shape)
    """
    Note that the outputs of 🤗Transformers models behave like namedtuples or dictionaries. You can access the
    elements by attributes (like we did) or by key (outputs["last_hidden_state"]), or even by index if you know exactly
    where the thing you are looking for is (outputs[0]).
    """


    # Model head
    # The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension.
    # For our example, we will need a model with a sequence classification head (to be able to classify the sentences as
    # positive or negative)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  # model head for classification
    
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    list_pred = predictions.tolist()

    col_list = []
    for i in range(len(model.config.id2label)):
        col_list.append(model.config.id2label[i])
    
    df_result = pd.DataFrame(list_pred, columns=col_list)
#    df_result.style.hide(axis="index")
    df_result=df_result.style.format(":,.2%")
#    df_result.style.highlight_min()
    print("dataframe", df_result)
    
    
    if in_detailed_log is True:
        print("Inputs: ", inputs)
        print("Model: ", model)
        print(outputs.logits.shape)
        # We have just 2 sentences and two labels, the result we get from our model is of shape 2 x 2.
        print(outputs.logits) # the raw, unnormalized scores outputted by the last layer of the model
        print(predictions)
        print(model.config.id2label)
    
    for sentence_no in range(outputs.logits.shape[0]):
        print("\nProbability for a {} sentence is {:.1f} %. Probability for a {} sentence is {:.1f} %.".format(
            model.config.id2label[0], predictions[sentence_no][0]*100,
            model.config.id2label[1], predictions[sentence_no][1]*100))

# MAIN part

# Let user select a script to be executed
option = selectFromDict(options, 'option')

if option == "posneg":
    sentence_sentiment_classification()
elif option == "ner":
    print("Nothing to show, YET!")
elif option == "quan":
    print("Nothing to show, YET!")
elif option == "summ":
    print("Nothing to show, YET!")
elif option == "transdero":
    print("Nothing to show, YET!")
elif option == "textgen":
    print("Nothing to show, YET!")

    