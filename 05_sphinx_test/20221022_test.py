# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Stefan W. Moeller
# Created Date: 2022/09/18
# version ='1.1'
# ----------------------------------------------------------------------------
"""
Purpose
--------
BERT stands for “Bidirectional Encoder Representation with Transformers”. To put it in simple words BERT extracts
patterns or representations from the data or word embeddings by passing it through an encoder. The encoder itself is
a transformer architecture that is stacked together. It is a bidirectional transformer which means that during
training it considers the context from both left and right of the vocabulary to extract patterns or representations.

BERT (Bidirectional Encoder Representation with Transformers) that achieved state-of-the-art performance in tasks
like Question-Answering, Natural Language Inference, Classification, and General language understanding evaluation
or (GLUE).

BERT accepts 1 or 2 sentences as input, not more!

-- input: 1-2 sentences as input only: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#2-input
-formatting

The entire program is broken down into 4 sections:

-- Preprocessing

-- Building model

-- Loss and Optimization

-- Training



1. The transformer concept
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-- source: https://jalammar.github.io/illustrated-transformer/


References
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following references were taken:

-- Main tutorial: https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial


=================
Used functions
=================

"""

# Importing packages and modules
# ----------------------------------------------------------------------------
import re



def printing(IN_text):
    """
    A sub function to print something
    
    :param IN_text: text
    :type IN_text: str
    :return: None
    :rtype: str
    
    """
    print(IN_text)
    return

def printing1(IN_text, In_text2 = "test"):
    """
    A sub function1 to print something
    
    :param In_text2: testenene
    :type In_text2: str
    :param IN_text: text
    :type IN_text: str
    :return:
    :rtype:
    """
    print(IN_text)
# TODO Check the code 1212
    return


if __name__ == "__main__":
    printing("This is a test")
# TODO Check the code
