from parse_configuration import Configuration
from collections import Counter

class ArcEager:
    def get_transition_from_parse_config(self, config, example):
        """ Used to generate training instances """
        if is_final_state(config):
            return None

        stack = config.stack
        buf = config.buffer
        sent = config.sentence 

        if len(buf)==0:
            return 'REDUCE'

        s0 = stack[-1]
        b0 = buf[0]
        head_s0 = example['head'][s0]
        head_b0 = example['head'][b0]

        if head_b0 == s0:
            return 'RIGHTARC' + '#' + example['label'][b0]
        if head_s0 == b0:
            return 'LEFTARC' + '#' + example['label'][s0]
        if head_s0 not in buf and len(get_children_from_buffer(s0,buf,example)) == 0:
            return 'REDUCE'
        else:
            return 'SHIFT'

    def update_state(self, config, transition):
        """ update config based on transition if valid """
        if len(config.buffer) == 0 and not transition == 'REDUCE':
            return config
        if transition == 'REDUCE':
            config.stack = config.stack[:-1]
            return config

        s0 = config.stack[-1]
        b0 = config.buffer[0]
        if transition == 'SHIFT':
            config.stack.append(b0)
            config.buffer = config.buffer[1:]
            return config

        label = transition.split('#')[-1]
        if transition.startswith('RIGHTARC'):
            config.relations.append((s0 , b0, label))
            config.stack.append(b0)
            config.buffer = config.buffer[1:]
        elif transition.startswith('LEFTARC'):
            config.relations.append((b0, s0, label))
            config.stack = config.stack[:-1]
        return config

    def map_transitions_to_id(self, train_set, unlabeled = False):
        root_labels = list([l for ex in train_set for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:
            print('Warning: more than one root label', counter)
        root_label = counter.most_common()[0][0]
        dep_rel = list(set([l for ex in train_set for l in ex['label'] if l not in ['L_ROOT', root_label]]))
        print(len(dep_rel), ' labels : ', dep_rel)

        transitions = ['SHIFT', 'REDUCE']
        if unlabeled:
            transitions += ['LEFTARC', 'RIGHTARC']
        else:
            transitions += ['LEFTARC#' + l for l in dep_rel] + ['RIGHTARC#' + l for l in dep_rel]
            transitions += ['RIGHTARC#' + root_label] # Only right arc is possible for root label

        transition_to_ix = {transition : i for i,transition in enumerate(transitions)}
        return transition_to_ix


    def get_legal_labels(self, stack, buf, relations, transition_size):
        """ The labels are ordered as shift, reduce, leftarc.., rightarc.., rightarc-root as defined in map_transitions_to_id function.
        Returns an array of transition_size containing boolean values to specify if transition at that index is valid."""
        n_dep_rels = int((transition_size - 2)/2)
        labels = [1] if len(buf) > 0 and not len(stack) == len(buf) == 1 else [0] #SHIFT

        attached = False
        if len(stack) > 1:
            for r in relations:
                if r[1] == stack[-1]:
                    attached = True # incoming arc already exists for this node. Cannot have another
            
        labels += [1] if attached else [0] #REDUCE
        labels += ([1] if len(stack) > 1 and len(buf) > 0 and not attached else [0]) * n_dep_rels #LEFTARC

        labels += ([1] if len(stack) > 1 and len(buf) > 0 else [0]) * n_dep_rels #RIGHTARC
        labels += [1] if stack[-1] == 0 and len(buf) > 0 else[0] #ROOT
        return labels

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
