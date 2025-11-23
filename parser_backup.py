import time
import os
import logging
from collections import Counter
from collections import defaultdict
from datetime import datetime
from config import Config
from arc_eager import ArcEager
from arc_standard import ArcStandard
from parse_configuration import Configuration
from parser_model import ParserModel
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Parser:

    def __init__(self, train_set, dev_set, bert_vec_dim=0):
        self.config = Config()
        self.train_set = train_set
        self.dev_set = dev_set
        self.parse_algo = ArcEager()
        if self.config.parsing_algo =='standard':
            self.parse_algo = ArcStandard()
        self.transition_to_ix = self.parse_algo.map_transitions_to_id(train_set, self.config.unlabeled)
        print("Loading pretrained embeddings...",)
        """ keyedVectors are a mapping from word to its corresponding vector """
        self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(self.config.embedding_file)
        print('Word embeddings size : ', len(self.word_vectors.vectors), ',' ,self.word_vectors.vector_size)
        # Add vectors for ROOT and null. A null verb or cc is added for a few sentences which do not have a verb.
        self.word_embedding_size = self.word_vectors.vector_size
        self.word_vectors.add(['ROOT', 'null'], np.random.normal(0, 0.9, (2, self.word_embedding_size)))

        self.pos_embeddings = gensim.models.KeyedVectors.load_word2vec_format(self.config.pos_embedding_file)
        self.pos_embedding_size = self.pos_embeddings.vector_size
        self.pos_embeddings.add(['P_ROOT', 'null'], np.random.normal(0, 0.9, (2, self.pos_embedding_size)))

        if self.config.use_bert:
            self.bert_vec_dim = bert_vec_dim
            print('BERT vector dimension ', self.bert_vec_dim)
            self.bert_vec_null = np.random.normal(0, 0.9, self.bert_vec_dim) # Used for padding when buffer is empty etc.

    def get_training_instances(self):
        training_instances = []
        for i, example in enumerate(self.train_set):
            parse_config = Configuration(example['word'], example['pos'], example['bert_vectors'] if self.config.use_bert else [])
            sentence_instances = []
            while(is_final_state(parse_config) == False):
                transition = self.parse_algo.get_transition_from_parse_config(parse_config, example)
                if(transition == None):
                    break
                sentence_instances.append((parse_config.get_state(), transition))
                """ update config based on transition if valid """
                parse_config = self.parse_algo.update_state(parse_config, transition)
            training_instances += sentence_instances
            # TODO: evaluate final config of the sentence to verify training oracle accuracy
        return training_instances

    def parse(self, model, data_set, eval_batch_size=500, write_to_conll=False, output_path=''):
        """print(80*'-')
        print('MODEL')
        print(model)
        for param in model.parameters():
            print(param.data.size())
            print('{0:.6f}'.format(torch.sum(param.data)))
        print(80*'-')"""
        parse_states = [Configuration(example['word'], example['pos'], example['bert_vectors'] if self.config.use_bert else []) for example in data_set]
        unfinished_parses = parse_states[:]
        ix_to_transition = {i : transition for transition, i in self.transition_to_ix.items()}
        
        while(len(unfinished_parses) > 0):
            minibatch = unfinished_parses[:eval_batch_size]
            debug_file.write('predicted_label|' +'\t'+ 'stack|'+ '\t'+'buffer|'+'\t'+'sentence'+'\n')
        
        while(len(unfinished_parses) > 0):
            minibatch = unfinished_parses[:eval_batch_size]
            minibatch_features = self.get_minibatch_features(minibatch)#All relevant vectors must be initialized in init.
            transitions = self.predict_transition(model, minibatch, minibatch_features, len(self.transition_to_ix))
#            print("Transitions : ", [ix_to_transition[i] for i in transitions])
            delete_indices = []
            for idx,parse_state in enumerate(minibatch):
                if debug:
                    debug_file.write(ix_to_transition[transition_idx] + '|\t' + ', '.join(parse_state.stack) + '|\t' + ', '.join(parse_state.buffer) + '|\t' + ' '.join(parse_state.sentence) + '\n')
                transition_idx = transitions[idx]
                self.parse_algo.update_state(parse_state, ix_to_transition[transition_idx])
            for idx,parse_state in enumerate(minibatch):
                if self.config.parsing_algo == 'eager' and len(parse_state.buffer)==0: 
                    # buffer being empty is a terminal state. Reduce operations here do not change anything
                    delete_indices.append(idx)
                elif is_final_state(parse_state):
                    delete_indices.append(idx)
            for idx in sorted(delete_indices, reverse=True):
                del unfinished_parses[idx]
        if write_to_conll:
            self.write_to_conll_file(parse_states, data_set, output_path)

        UAS,LAS,LA = self.evaluate(parse_states, data_set)
        return UAS, LAS, LA

    def write_to_conll_file(self, parsed_examples, data_set, output_path):
        gold_f = open(output_path+'_gold.conll', 'w')
        predicted_f = open(output_path + '_predicted.conll', 'w')
        for i, example in enumerate(data_set):
            sent = defaultdict(list)
            p_sent = defaultdict(list)
            predicted_relations = parsed_examples[i].relations

            for idx, word in enumerate(example['word']):
                tokens = [str(idx), word] + 4*['_'] + [str(example['head'][idx]), example['label'][idx]] + 2*['_']
                p_tokens = [str(idx), word] + 8*['_']
                sent[idx] = tokens
                p_sent[idx] = p_tokens
            for relation in predicted_relations:
                p_sent[relation[1]][6] = str(relation[0])
                p_sent[relation[1]][7] = relation[2]
            for line_x in range(1, len(sent)):
                gold_f.write('\t'.join(sent[line_x]) + '\n')
                predicted_f.write('\t'.join(p_sent[line_x]) + '\n')
            gold_f.write('\n')
            predicted_f.write('\n')


    def get_minibatch_features(self, minibatch):
        features = []
        for i,parse_state in enumerate(minibatch):
            feature_vec = self.create_feature_vector(parse_state.get_state())
            features.append(feature_vec)
        features_tensor = torch.tensor(np.asarray(features, dtype='float32')) # In pytorch, all tensors of default type float32.
        return features_tensor


    def predict_transition(self, model, minibatch, minibatch_features, n_transitions):
        legal_labels = [self.parse_algo.get_legal_labels(parse_state.stack, parse_state.buffer,parse_state.relations, n_transitions) for parse_state in minibatch] #size:(batch_size, n_transitions)
        probs = model(minibatch_features)
        probs = probs.detach().numpy()
        transitions = np.argmax(probs + 10000*np.array(legal_labels).astype('float32'), 1)
        return transitions
            
    def evaluate(self, parsed_examples, data_set):
        """ Evaluate the parsed sentences. """
        UAS = LAS = LA = n_tokens = 0
        for i,example in enumerate(data_set):
            predicted_relations = parsed_examples[i].relations
            n_tokens += len(example['word'])-1
            if not self.config.parsing_algo == 'eager': # Arc-eager algo does not guarantee fully formed trees
                assert len(predicted_relations) == len(example['word']) - 1
            for relation in predicted_relations:
                head_match = example['head'][relation[1]] == relation[0]
                label_match = example['label'][relation[1]] == relation[2]
                UAS += 1 if head_match else 0
                LA += 1 if label_match else 0
                LAS += 1 if head_match and label_match else 0
        UAS = UAS/n_tokens
        LAS = LAS/n_tokens
        LA = LA/n_tokens
#        TODO write to output file in conll format
        return UAS, LAS, LA
        

    def get_minibatches(self, data, minibatch_size=1024, shuffle=True):
        data_size = len(data)
        indices = np.arange(data_size)
        minibatches = []
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, data_size, minibatch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
            features = []
            target = []
            for i,idx in enumerate(minibatch_indices):
                instance = data[idx]
                feature_vec = self.create_feature_vector(instance[0])
                features.append(feature_vec)
                target.append(self.transition_to_ix[instance[1]])
            features_tensor = torch.tensor(np.asarray(features, dtype='float32')) # In pytorch, all tensors of default type float32.
            target_tensor = torch.LongTensor(target)
            minibatches.append((features_tensor, target_tensor))
        return minibatches

    def create_model(self):
        feature_vec_size = self.word_embedding_size
        if self.config.use_bert:
            feature_vec_size = feature_vec_size + self.bert_vec_dim if self.config.concat_bert_with_word_vec else self.bert_vec_dim
        if self.config.use_pos:
            feature_vec_size += self.pos_embedding_size

        n_features = 6 if self.config.feature_template else 2
        n_features += 1 if self.config.parsing_algo == 'eager' else 0

        feature_size = n_features * feature_vec_size
        label_size = len(self.transition_to_ix)
        print('Feature size, Label size (', feature_size, ',', label_size, ')')
        model = ParserModel(feature_size, label_size, hidden_size=2048 if self.config.use_bert else 256)
        return model


    def train(self, model, output_path, n_epochs=50):
        """ Each instance is of format ({'stack': stack, 'buffer': buf, 'relations': relations, 'sentence': sentence}, transition) """

        print("Generating training instances...")
        training_instances = self.get_training_instances()
        print('Done!')

        print('Training parser....')
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
#        optimizer = optim.Adagrad(model.parameters(), weight_decay=0.003)
        model.train()
        best_dev_UAS = 0
        best_dev_LAS = 0
        for epoch in range(n_epochs):
            print('Epoch : ', epoch+1)
            for i, (features, target) in enumerate(self.get_minibatches(training_instances)):
#                model.zero_grad()
                optimizer.zero_grad()
                log_probs = model(features)
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()

            dev_UAS, dev_LAS, dev_LA = self.parse(model, self.dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
                print("- dev LA: {:.2f}".format(dev_LA * 100.0))
                print("- dev LAS: {:.2f}".format(dev_LAS * 100.0))
                print("New best dev UAS! Saving model.")
                torch.save(model.state_dict(), output_path)


    def create_feature_vector(self, parse_state):
        if self.config.feature_template:
            return self.create_template_feature_vector(parse_state)

        s0 = parse_state['stack'][0]
        sentence = parse_state['sentence']

        s0_vec = look_up_keyed_embedding(sentence[s0], self.word_vectors)
        if self.config.use_bert:
            if self.config.concat_bert_with_word_vec:
                s0_vec = np.concatenate((parse_state['bert_vectors'][s0], s0_vec))
            else:
                s0_vec = parse_state['bert_vectors'][s0]

        if not parse_state['buffer']:
            b0_vec = self.word_vectors.get_vector('null')
            if self.config.use_bert:
                if self.config.concat_bert_with_word_vec:
                    b0_vec = np.concatenate((self.bert_vec_null, b0_vec))
                else:
                    b0_vec = self.bert_vec_null
        else:
            b0 = parse_state['buffer'][0]
            b0_vec = look_up_keyed_embedding(sentence[b0], self.word_vectors)
            if self.config.use_bert:
                if self.config.concat_bert_with_word_vec:
                    b0_vec = np.concatenate((parse_state['bert_vectors'][b0], b0_vec))
                else:
                    b0_vec = parse_state['bert_vectors'][b0]
                       
        feature_vector = np.concatenate((s0_vec, b0_vec))
        return feature_vector


    def create_template_feature_vector(self, parse_state):
        #(word, pos features for s0, s1, s2, s3, b0, b1, b2, b3, lc1(s0), rc1(s0), lc1(s1), rc1(s1), lc1(b0)) -> n_stack=4, n_buf=4, n_schildren=2
        #Template2 : s0, s1, s2, b0, lc1(s0), rc1(s0) -> n_stack=3, n_buf=1, n_schildren=1
        sentence = parse_state['sentence']
        pos_sentence = parse_state['pos']
        stack = parse_state['stack']
        buf = parse_state['buffer']
        relations = parse_state['relations']

        n_stack = 3 #( s0, s1, s2 )
        n_buf = 1 #( b0 )
        n_schildren = 1 #( lc1(s0), rc1(s0) )

        word_vec_null = self.word_vectors.get_vector('null')
        if self.config.use_bert:
            word_vec_null = np.concatenate((self.bert_vec_null, word_vec_null)) if self.config.concat_bert_with_word_vec else self.bert_vec_null
        pos_vec_null = self.pos_embeddings.get_vector('null')

        pos_vecs=[]
        word_vecs = [word_vec_null] * (n_stack-len(stack)) + \
                [self.look_up_embedding(sentence[i], self.word_vectors, i, parse_state) for i in stack[-1*n_stack:]]
        word_vecs += [self.look_up_embedding(sentence[i], self.word_vectors, i, parse_state) for i in buf[:n_buf]] + [word_vec_null] * (n_buf-len(buf))

        if self.config.use_pos:
            pos_vecs = [pos_vec_null] * (n_stack-len(stack)) + [look_up_keyed_embedding(pos_sentence[i], self.pos_embeddings) for i in stack[-1*n_stack:]]
            pos_vecs += [look_up_keyed_embedding(pos_sentence[i], self.pos_embeddings) for i in buf[:n_buf]] + [pos_vec_null] * (n_buf-len(buf))
        
        #child features
        for i in range(n_schildren):
            if len(stack) > i:
                k = stack[-i-1]
                lc = self.get_lc(k, relations)
                rc = self.get_rc(k, relations)
                    pos_vecs.append(look_up_keyed_embedding(pos_sentence[rc[0]], self.pos_embeddings) if len(rc) > 0 else pos_vec_null)

            else:
                word_vecs += [word_vec_null] * 2
                if self.config.use_pos:
                    pos_vecs += [pos_vec_null] * 2

        #lc1(b0) only for arc_eager
        if self.config.parsing_algo == 'eager':
            lc_b0 = self.get_lc(buf[0], relations) if len(buf) > 0 else []
            word_vecs += [self.look_up_embedding(sentence[lc_b0[0]], self.word_vectors, lc_b0[0], parse_state) if len(lc_b0) > 0 else word_vec_null]
            if self.config.use_pos:
                pos_vecs += [look_up_keyed_embedding(pos_sentence[lc_b0[0]], self.pos_embeddings) if len(lc_b0) > 0 else pos_vec_null]

        features = word_vecs + pos_vecs 
        feature_vector = np.concatenate((features))
        return feature_vector

    def get_lc(self, k, relations): # get left children
        return sorted([arc[1] for arc in relations if arc[0] == k and arc[1] < k])

    def get_rc(self, k, relations): # get right children
        return sorted([arc[1] for arc in relations if arc[0] == k and arc[1] > k], reverse=True)

    def look_up_embedding(self, key, keyed_vectors, index, parse_state):
        embedding = look_up_keyed_embedding(key, keyed_vectors)
        if self.config.use_bert:
            bert_vec = parse_state['bert_vectors'][index]
            embedding = np.concatenate((bert_vec, embedding)) if self.config.concat_bert_with_word_vec else bert_vec
        return embedding



def look_up_keyed_embedding(key, keyed_vectors):
    try:
        embedding = keyed_vectors.get_vector(key) #returns np.ndarray
    except KeyError: 
        embedding = np.random.normal(0, 0.9, keyed_vectors.vector_size)
        keyed_vectors.add([key], [embedding])
    return embedding


def is_final_state(parse_config):
    if parse_config.stack == [0] and parse_config.buffer == []:
        return True
    else:
        return False


def get_children_from_buffer(s0, buf, example):
    children = [] 
    for word in buf:
        if example['head'][word] == s0:
            children.append(word)
    return children



