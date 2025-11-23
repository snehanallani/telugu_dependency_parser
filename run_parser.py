from datetime import datetime
import os
import pickle
import math
import time
import json

import gensim
from torch import nn, optim
import torch
from parser_model import ParserModel
from config import Config
from collections import defaultdict
from parser import *

"""
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('parsing_algo', 'eager', 'can be eager or standard')
flags.DEFINE_boolean('feature_template', True, ' Uses Template (word, pos features for s0, s1, s2, s3, b0, b1, b2, b3, lc1(s0), rc1(s0), lc1(s1), rc1(s1), lc1(b0))')
flags.DEFINE_boolean('use_bert', True, '')
flags.DEFINE_list('bert_layers', [-1, -2, -3, -4], 'Specify which layers of bert to be used')
flags.DEFINE_string('mode', 'add', 'Specify whether to add or concat BERT layers')
"""


def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': ['ROOT'] + word, 'pos': ['P_ROOT'] + pos, 'head': [-1] + head, 'label': ['L_ROOT'] + label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': ['ROOT'] + word, 'pos': ['P_ROOT'] + pos, 'head': [-1] + head, 'label': ['L_ROOT'] + label})
    return examples


def remove_non_projective_sentences(data_set):
    #print('Non-projective sentences...')
    non_proj_indices = []
    for i, example in enumerate(data_set):
        if is_non_projective(example):
            non_proj_indices.append(i)
    for idx in sorted(non_proj_indices, reverse=True):
        #print(idx)
        #print(data_set[idx]['word'], data_set[idx]['head'], data_set[idx]['label'])
        del data_set[idx]
    return data_set

def is_non_projective(example):
    """ Checks if the given sentence has a non-projective tree """
    edges = []
    tree = defaultdict(list)
    words = example['word']
    heads = example['head']
    for i in range(1,len(words)):
        tree[heads[i]].append(i)
        edges.append((heads[i],i))

    for edge in edges:
        if edge[0]-edge[1] < -1: # right-arc
            for i in range(edge[0]+1, edge[1]):
                if not path_exists(edge[0], i, tree):
                    return True
        elif edge[0]-edge[1] > 1: #left-arc
            for i in range(edge[0]-1, edge[1], -1):
                if not path_exists(edge[0], i, tree):
                    return True
    return False


def path_exists(source, target, tree):
    """ Given a dependency tree, checks if a path exists between the source and target nodes in the tree. 
    The source and target nodes are indices of the words in the sentence"""
    nodes_to_visit = [source]
    while nodes_to_visit:
        node = nodes_to_visit[0]
        for child in tree[node]:
            if child == target:
                return True
            nodes_to_visit.append(child)
        nodes_to_visit = nodes_to_visit[1:]
    return False

def write_sentences_to_file(file_name, example_set):
    f1 = open(file_name, 'w')
    for example in example_set:
        f1.write(" ".join(example['word'])+'\n')        

def read_json_line(line, layers_to_use, mode):
    # extract all tokens in a sentence and their vectors (sum of all available layers for now).
    tokens = []
    vectors = []
    data = json.loads(line)
    line_num = data['linex_index']
    token_features = data['features']
    for token_data in token_features:
        token = token_data['token']
        layer_vecs = {}
        json_layers = token_data['layers']
        for layer in json_layers:
            idx = layer['index']
            vec = layer['values']
            layer_vecs[idx]=vec
        tokens.append(token)
        temp = []
        for l in layers_to_use:
            temp.append(layer_vecs[l])
        if mode == 'concat':
            vectors.append(np.concatenate((temp)))
        else: # mode == 'add'
            vectors.append(np.sum(temp, axis=0))

    return line_num, tokens, vectors


def merge_word_tokens_and_vectors(tokens, vectors):
    # merge all the tokens of words and add their corresponding vectors.
    word_ids = []
    words = []
    word_vectors = []
    for ix, token in enumerate(tokens):
        if not token.startswith('##'):
            word_ids.append(ix)
    for i in range(1, len(word_ids)-1): #ignore [CLS] and [SEP] tokens
        cur_id = word_ids[i]
        next_id = word_ids[i+1]
        word_ = tokens[cur_id]
        vec_ = vectors[cur_id]
        if next_id - cur_id > 1:
            for j in range(cur_id+1,next_id):
                word_ += tokens[j][2:]
                vec_ += vectors[j]
        words.append(word_)
        word_vectors.append(vec_)
    return words, word_vectors


def correct_compound_word_case(bert_words, data_set, sent_num, vectors):
    sentence_words = data_set[sent_num]['word']
    for i, bert_word in enumerate(bert_words):
        if len(sentence_words) > i:
            sent_word =  sentence_words[i]
            if bert_word != sent_word:
#                print(20*'-', bert_word, sent_word)
                j = i+1
                temp_vec = vectors[i]
                while(j < len(bert_words)):
                    bert_word += bert_words[j]
                    temp_vec += vectors[j]
                    if bert_word == sent_word:
                        bert_words[i] = bert_word
                        vectors[i] = temp_vec
                        del bert_words[i+1:j+1]
                        del vectors[i+1:j+1]
                        if bert_words == sentence_words:
                            data_set[sent_num]['bert_vectors'] = vectors
                            print('Corrected compound word case in sentence ', bert_word, sent_num)
                            return True
                        break
                    j += 1
    return False


def parse_bert_json(data_set, bert_json_file, layers_to_use, mode):
    f = open(bert_json_file, 'r')
    lines = f.readlines()
    mismatched_ids = []
    for idx, line in enumerate(lines):
        line_num, tokens, vectors = read_json_line(line, layers_to_use, mode)
        assert line_num == idx
        words, word_vectors  = merge_word_tokens_and_vectors(tokens, vectors)

        #assert data_set[idx]['word'] == words
        #data_set[idx]['bert_tokens'] = words
        if not data_set[idx]['word'] == words:
            #TODO Remove the mismatched sentences for now. Handle NULL_VGF cases. Correct the treebank sentences in other cases.
            if not correct_compound_word_case(words, data_set, idx, word_vectors):
                mismatched_ids.append(idx)
                print(data_set[idx]['word'])
                print(words)
        else:
            data_set[idx]['bert_vectors'] = word_vectors
    mismatched_ids.sort(reverse = True)
    for ix in mismatched_ids:
        del data_set[ix]

def print_config(config):
    print('Use POS: ', config.use_pos)
    print('Parsing Algo: ', config.parsing_algo)
    print('Use BERT: ', config.use_bert)
    print('BERT Layers: ', config.bert_layers)
    print('BERT layers mode: ', config.mode)
    print('Concatenate BERT with word_vec: ', config.concat_bert_with_word_vec)
    #print('Inverse BPE: ', 'tel_inverse_bpe_model' in config.bert_train_file_vectors)


if __name__ == "__main__":

    time_str = datetime.now().strftime("%I:%M%p-%B%d,%Y")
    print(time_str)
    print(80 * '=')
    config = Config()
    print_config(config)
    train_set = read_conll(os.path.join(config.data_path, config.train_file), lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file), lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file), lowercase=config.lowercase)

    #Get all the train, dev and test set sentences and extract ELMo like fixed vectors from bert model for these sentences.
    """write_sentences_to_file('train_sentences.txt', train_set)
    write_sentences_to_file('dev_sentences.txt', dev_set)
    write_sentences_to_file('test_sentences.txt', test_set)"""

    if config.use_bert:
        print('Loading BERT vectors .....')
        parse_bert_json(train_set, config.bert_train_file_vectors, config.bert_layers, config.mode)
        parse_bert_json(dev_set, config.bert_dev_file_vectors, config.bert_layers, config.mode)
        parse_bert_json(test_set, config.bert_test_file_vectors, config.bert_layers, config.mode)
        print('Done!')

    train_set = remove_non_projective_sentences(train_set)
    dev_set = remove_non_projective_sentences(dev_set)
    test_set = remove_non_projective_sentences(test_set)
   
    print('Total training sentences : ', len(train_set))
    print('Total dev sentences : ', len(dev_set))
    print('Total test sentences : ', len(test_set))
    #print('Example : ', train_set[0])
    print(80 * '=')

    parser = Parser(train_set, dev_set, bert_vec_dim = len(train_set[0]['bert_vectors'][0]) if config.use_bert else 0)
    if config.train_parser == True:
        output_dir = config.output_dir
        output_path = output_dir + time_str + "model.weights"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model = parser.create_model()
        parser.train(model,output_path,n_epochs=24)

        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set with Arc", config.parsing_algo, " algorithm")
        model.eval()
        test_UAS, test_LAS, test_LA = parser.parse(model, test_set, write_to_conll=True, output_path=output_dir + 'out_'+time_str, debug=False)
        print("- test UAS: {:.2f}".format(test_UAS * 100.0))
        print("- test LA: {:.2f}".format(test_LA * 100.0))
        print("- test LAS: {:.2f}".format(test_LAS * 100.0))
    else:
        #TODO load model from path and evaluate on test set
        print('****')
