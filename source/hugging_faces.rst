Working with pipelines
----------------------
Different utilizations of models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Sentiment analysis
^^^^^^^^^^^^^^^^^^^^^^
Classifying whole sentences: Getting the sentiment of a review, detecting if an email is spam, determining if a
sentence is grammatically correct or whether two sentences are logically related or not

2. feature-extraction
^^^^^^^^^^^^^^^^^^^^^^^
get the vector representation of a text

3. fill-mask
^^^^^^^^^^^^^
The idea of this task is to fill in the blanks in a given text.

4. ner (named entity recognition)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to
entities such as persons, locations, or organizations.

5. question-answering
^^^^^^^^^^^^^^^^^^^^^^
The question-answering pipeline answers questions using information from a given context
Note that this pipeline works by extracting information from the provided context; it does not generate the answer.

6. summarization
^^^^^^^^^^^^^^^^^
Summarization is the task of reducing a text into a shorter text while keeping all (or most) of the important aspects
referenced in the text.

7. text-generation
^^^^^^^^^^^^^^^^^^^
The main idea here is that you provide a prompt and the model will auto-complete it by generating
the remaining text. This is similar to the predictive text feature that is found on many phones.

8. translation
^^^^^^^^^^^^^^^
For translation, you can use a default model if you provide a language pair in the task name (such as
"translation_en_to_fr"), but the easiest way is to pick the model you want to use on the Model Hub.

9. zero-shot-classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We’ll start by tackling a more challenging task where we need to classify texts that haven’t been labelled.
You’ve already seen how the model can classify a sentence as positive or negative using those two labels — but it can
also classify the text using any other set of labels you like.


NLP models
----------
This list is far from comprehensive, and is just meant to highlight a few of the different kinds of Transformer models. Broadly, they can be grouped into three categories:

- GPT-like (also called auto-regressive Transformer models)

- BERT-like (also called auto-encoding Transformer models)

- BART/T5-like (also called sequence-to-sequence Transformer models)

Encoder model
^^^^^^^^^^^^^^
Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the
words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are
often called auto-encoding models.

The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking
random words in it) and tasking the model with finding or reconstructing the initial sentence.

Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence
classification, named entity recognition (and more generally word classification), and extractive question answering.

Decoder model
^^^^^^^^^^^^^^
Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can
only access the words positioned before it in the sentence. These models are often called auto-regressive models.

The pretraining of decoder models usually revolves around predicting the next word in the sentence.

These models are best suited for tasks involving text generation.

Encoder-decoder models
^^^^^^^^^^^^^^^^^^^^^^^^
Encoder-decoder models (also called sequence-to-sequence models) use both parts of the Transformer architecture. At
each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.

The pretraining of these models can be done using the objectives of encoder or decoder models, but usually involves
something a bit more complex. For instance, T5 is pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces.

Sequence-to-sequence models are best suited for tasks revolving around generating new sentences depending on a given
input, such as summarization, translation, or generative question answering.

Terminology
^^^^^^^^^^^^
- Architecture: This is the skeleton of the model — the definition of each layer and each operation that
happens within the model.

- Checkpoints: These are the weights that will be loaded in a given architecture.

- Model: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both. This
course will specify architecture or checkpoint when it matters to reduce ambiguity.

