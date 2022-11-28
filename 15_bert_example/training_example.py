"""
Purpose
===========

understand the meaning of ambiguous language in text by using surrounding text to establish context
The objective of Masked Language Model (MLM) training is to hide a word in a sentence and then have the program
predict what word has been hidden (masked) based on the hidden word's context. The objective of Next Sentence
Prediction training is to have the program predict whether two given sentences have a logical, sequential connection
or whether their relationship is simply random.

Example: Next word prediction in google search

The transformer does this by processing any given word in relation to all other words in a sentence, rather than
processing them one at a time. By looking at all surrounding words, the Transformer allows the BERT model to
understand the full context of the word, and therefore better understand searcher intent.

Approach in python file
-----------------------


1. Generate a list of words from cleaned sentences
Removing the punctuations from the sentences and generate a list of unique words.



Concept
-------
word embedding
^^^^^^^^^^^^^^^
The word -embeddings- is represented:
['em', '##bed', '##ding', '##s']
The original word has been split into smaller subwords and characters. This is because Bert Vocabulary is fixed with
a size of ~30K tokens. Words that are not part of vocabulary are represented as subwords and characters.


Some definition
^^^^^^^^^^^^^^^^
word embedding: feature vector representation of a word



Reference
^^^^^^^^^^
background: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model
background BERT: https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca

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


# Classes and functions and more

def generate_clean_sentence(in_text: str):
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

def generate_dictionary(IN_word_list: list):
    """
    Out of a provided list of unique words, generate two dictionaries, one with word to number and one with number
    to word relation.
    
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
    Generate a list containing each sentence as a single item. Convert the individual sentences into their tokens,
    i.e. convering the words into numbers using the word dictionary (obmitting the special tokens)
    
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


def make_batch(IN_sentences: list, batch_size: int = 84, IN_max_pred: int = 3, IN_maxlen: int = 23):
    """
    Select two random sentences out of the list of tokeneized sentences.
    
    1) check the number of sentences and select two random token index (used to select tokenized sentences)
    2) Using the random index, select two (already as number converted sentences)
    3) we add special tokens to the sentences
    4) represent the first sentence and the second sentence as a sequence of "0" and "1"
    
    :param IN_sentences: List, containing the sentences (cleaned from punctuations, etc.)
    :type IN_sentences: lst
    :param batch_size: Size of each batch
    :type batch_size: int
    :param IN_max_pred:
    :type IN_max_pred:
    :param IN_maxlen:
    :type IN_maxlen:
    :return:
    :rtype:
    """
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
 
        # 1) Select two indexes out of the range of contained sentences in the list "sentences"
# ToDo Warum wird die "sentence" Liste zur Auswahl genutzt und nicht die "token" Liste. ANTWORT: IST EGAL
        # The randrange() method returns a randomly selected element from the specified range.
        tokens_a_index, tokens_b_index = randrange(len(IN_sentences)), randrange(len(IN_sentences))
       
        # 2) Using the indexes, select two (already as number converted) sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        print("\n[INFO] Random token A [{}] and B [{}] were randomly generated and used to derive the sentence A: {} "
              "and B: {}".format(tokens_a_index,tokens_b_index, tokens_a, tokens_b))
        
        # 3) we add special tokens to the sentences
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        print("[INFO] <input_ids>: Special tokens were added and both sentenced concatenated: ", input_ids)
    
        # 4) represent the first sentence and the second sentence as a sequence of "0" and "1"
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        print("[INFO] <segment_ids> - segmented sentence was build: ", segment_ids)
        
        # 5) Mask 15% of tokens in "one sentence", i.e. e.g. two sentences put together and cleaned from punctuations.
        # We mask at least one word and at most `max_pred` words.
        # 5a) Select the minimum out of a range of (max_pred = 3 (if 15% of <len of input ids> is greater than 3)) and
        # 1 (if 15% of len(input_ids) is less than 1))
        # ToDo: Why will there maximum be 3 words to be masked? ANTWORT: WHY NOT?
        
        n_pred = min(IN_max_pred, max(1, int(round(len(input_ids) * 0.15))))
        print("[INFO] <n_pred> - number of words to be masked: ", n_pred)
        
        # generate a list with length of the concatenated, tokenized sentence excl. special tokens, i.e. tokenized
        # sentence [1,3,5,2,4,3,2] becomes [excl. 0 as special token in dict, 1, 2, excl. special token, 4, 5,
        # excl. special token in dict]
        cand_masked_pos = [i for i, token in enumerate(input_ids) if
                           token != word_dict['[CLS]'] and token != word_dict['[SEP]']] # avoid counting the special
        # tokens
        print("[INFO] <cand_masked_pos> - list of increasing numbers (i.e. position of single words the position, "
              "where a special token is being used: ", cand_masked_pos)
        
        shuffle(cand_masked_pos)
        print("[INFO] <cand_masked_pos> Randomized list of word positions: ", cand_masked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_masked_pos[:n_pred]:    # from position 0 for number of words to be masked
            masked_pos.append(pos)              # generate a list with the position numbers to be masked
            print("[INFO] <masked_pos> - the positions, which will be masked ", masked_pos)
            masked_tokens.append(input_ids[pos])
            print("[INFO] <masked_pos> - from the input_ids and the positions (to be masked), lookup the word in the "
                  "<input_ids>: ", masked_tokens)
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # replace the number (i.e. word) in <input_ids> by the
                # respective number representing the token "mask" in case, the random is less than 80%
            elif random() < 0.5:  # 10% - wann wird denn dieses if gezogen?? 50% ist auch kleiner als 80% und wuerde
                # garnicht mehr geprueft werden
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]]  # replace based on number dict and word dict reversal
                # look-up
        print("[INFO] <input_ids> now reworked with masks words (in dict: number 3): ", input_ids)

        # Zero Paddings
        n_pad = IN_maxlen - len(input_ids)
        print("[INFO] <n_pad> Number of ZERO paddings to fill up the sentence up to the defined maximum length", n_pad)
        # ToDo What happens, if the length of the sentence is larger than the max length defined?
        input_ids.extend([0] * n_pad)       # add number of [PAD] to reach max len of sentence defined
        segment_ids.extend([0] * n_pad)

        print("[INFO] <segment_ids> New segment_ids after filling up the segment_ids to the max length required ",
              input_ids)
        # append the respective number of zeros in segment_ids
        # append the respective number of zeros in input_ids

        # Zero Padding (100% - 15%) tokens
        if IN_max_pred > n_pred:            # max words > words decided to be masked
            n_pad = IN_max_pred - n_pred    # delta between max mask words and decided words to be filled with [PAD]
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
# ToDo No clue, what's going on here...
        
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
        print("[INFO] <batch> Content of batch :", batch )
    return batch


# ## Now, the main program starts

if __name__ == "__main__":
    # Generate a list of sentences and single words out of a string of sentences
    sentences, word_list = generate_clean_sentence(text)
    print("[INFO] Generated a list of sentences and single words out of a string with sentences.")
    if deep_report is True:
        print("\n[DETAILED LOG] List of sentences cleaned by punctuations.\n", sentences)
        print("\n[DETAILED LOG] List of unique words.\n", word_list)
    
    word_dict, number_dict, vocab_size = generate_dictionary(word_list)
    print("\n[INFO] A word dictionary and number dictionary was generated and the length of the word dictionary "
          "determined.")
    if deep_report is True:
        print("\n[DETAILED LOG] The word dictionary.\n", word_dict)
        print("\n[DETAILED LOG] The number dictionary.\n", number_dict)
        print("\n[DETAILED LOG] Length of the word dictionary: ", vocab_size)
    
    token_list = convert_sentence_into_tokens(sentences, word_dict)
    print("[INFO] Token list generated, i.e. conversion of sentences into numbers: ", token_list)
  
    maxlen = 23
    batch = make_batch(IN_sentences=sentences, IN_maxlen=maxlen)
    
    batchentry = batch[0]
    input_ids = batchentry[0]
    
    model = BERT(maxlen=maxlen)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    batch = make_batch(IN_sentences=sentences)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))
    
    for epoch in range(20):
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
