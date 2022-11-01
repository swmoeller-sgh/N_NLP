"""
Purpose
===========

Some initial test for sphinx


Approach in python file
-----------------------
Test eiwe wies fsfs
fsfs
f
fsf

* ssfsfsf
    * sfsfs
    * sfsf
* sfsf

Generate a list of words from cleaned sentences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Removing the punctuations from the sentences and generate a list of unique words.

sdsdsdsd
^^^^^^^^^
sdsdll dsdlsdsd dslds

"""

import math
import re
from random import randrange, shuffle, random, randint

import torch
from torch import nn
from torch.optim import Adam

from bert import BERT

text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)
deep_report = True

########################
# Classes and functions
########################

def generate_clean_sentence (in_text: str):
    """
    1) loc_sentences = Generate a list with all sentences from one line, convert to lower case and remove punctuation.
    2) loc_word_list = One list containing every word of all sentences (without duplication)
    
    :param in_text: One string containing the text to be used for training
    :type in_text: str
    :return: sentences, word_list
    :rtype: list
    """
    loc_sentences = re.sub("[.,!?\\-]", '', in_text.lower()).split('\n')  # filter '.', ',', '?', '!'
    loc_word_list = list(set(" ".join(loc_sentences).split()))
    return loc_sentences, loc_word_list

def generate_dictionary (IN_word_list: list):
    """
    Out of a provided list of unique words, generate two dictionaries, one with word to number and one with number to word relation.
    
    [Token] | Purpose
    
    * [CLS] The first token is always classification
    * [SEP] Separates two sentences
    * [END] End the sentence.
    * [PAD] Use to truncate the sentence with equal length.
    * [MASK] Use to create a mask by replacing the original word.
    
    :param IN_word_list: List containing single words without punctuation, all in lower case
    :type IN_word_list: list
    :return: loc_word_dict, loc_number_dict, loc_vocab_size
    :rtype: dict, dict, int
    """
    
    loc_word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(IN_word_list):
        # The enumerated object yields pairs containing a count (from start, which defaults to zero) and a value yielded
        # by the iterable argument. Enumerate is useful for obtaining an indexed list:
        # (0, seq[0]), (1, seq[1]), (2, seq[2]), ...
        loc_word_dict[w] = i + 4
        loc_number_dict = {i: w for i, w in enumerate(loc_word_dict)}
        loc_vocab_size = len(loc_word_dict)
    return loc_word_dict, loc_number_dict, loc_vocab_size

  

def convert_sentence_into_tokens(IN_sentences, IN_word_dict):
    """
    Take each sentence and transform every word in the sentence in its specific token, i.e. generating a list
    containing a list of tokens
    
    :param IN_sentences:
    :type IN_sentences:
    :param IN_word_dict:
    :type IN_word_dict:
    :return:
    :rtype:
    """
    loc_token_list = []
    for sentence in IN_sentences:
        loc_token_list.append([IN_word_dict[word] for word in sentence.split()])
    return loc_token_list


def make_batch(batch_size: int = 84, max_pred: int = 3, maxlen: int = 23):
    """
    Select two random sentences out of the list of tokeneized sentences.
    
    1) check the number of sentences and select two random token index (used to select tokenized sentences)
    2) Using the random index, select two (already as number converted sentences)
    3) we add special tokens to the sentences
    4) represent the first sentence and the second sentence as a sequence of "0" and "1"
    
    :param batch_size:
    :type batch_size:
    :param max_pred:
    :type max_pred:
    :param maxlen:
    :type maxlen:
    :return:
    :rtype:
    """
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        # Select two random sentences out of the list of tokeneized sentences.
        # The randrange() method returns a randomly selected element from the specified range.
 
        # 1) check the number of sentences and select two random token index (used to select tokenized sentences)
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        
        # 2) Using the random index, select two (already as number converted sentences)
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        print("\n[INFO] Random token A [{}] and B [{}] were randomly generated and used to derive the sentence A: {} "
              "and B: {}".format(tokens_a_index,tokens_b_index, tokens_a, tokens_b))
        
        # 3) we add special tokens to the sentences
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        print("[INFO] Special tokens were added and both sentenced concatenated: ", input_ids)
    
        # 4) represent the first sentence and the second sentence as a sequence of "0" and "1"
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        print("[INFO] Segmented sentence was build : ", segment_ids)
        
        # MASK LM
        # 5) Mask 15% of tokens in one sentence. We mask at least one word and at most `max_pred` words.
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
        cand_masked_pos = [i for i, token in enumerate(input_ids) if
                           token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_masked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]]  # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    return batch


if __name__ == "__main__":
    # Generate a list of sentences and single words out of a string of sentences
    sentences, word_list = generate_clean_sentence(text)
    print("[INFO] Generated a list of sentences and single words out of a string with sentences.")
    if deep_report == True:
        print("\n[DETAILED LOG] List of sentences cleaned by punctuations.\n", sentences)
        print("\n[DETAILED LOG] List of unique words.\n", word_list)
    
    word_dict, number_dict, vocab_size = generate_dictionary(word_list)
    print("\n[INFO] A word dictionary and number dictionary was generated and the length of the word dictionary "
          "determined.")
    if deep_report == True:
        print("\n[DETAILED LOG] The word dictionary.\n", word_dict)
        print("\n[DETAILED LOG] The number dictionary.\n", number_dict)
        print("\n[DETAILED LOG] Length of the word dictionary: ", vocab_size)
    
    token_list = convert_sentence_into_tokens(sentences, word_dict)

    print("[INFO] Token list generated, i.e. conversion of sentences into numbers: ", token_list)
    maxlen = 23
    batch = make_batch(maxlen=maxlen)
    
    batchentry = batch[0]
    input_ids = batchentry[0]
    
    model = BERT(maxlen=maxlen)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    batch = make_batch()
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))
    
    for epoch in range(100):
        optimizer.zero_grad()
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext)  # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
    
    # Predict mask tokens
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
    print(text)
    print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])
    
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('masked tokens list : ', [pos.item() for pos in masked_tokens[0] if pos.item() != 0])
    print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])
    
    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('isNext : ', True if isNext else False)
    print('predict isNext : ', True if logits_clsf else False)
