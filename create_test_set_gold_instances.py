from arc_standard import ArcStandard
from parse_configuration import Configuration
from run_parser import *


def get_training_instances(data_set):
    training_instances = []
    algo = ArcStandard()
    for i, example in enumerate(data_set):
        parse_config = Configuration(example['word'], example['pos'], [])
        sentence_instances = []
        while(is_final_state(parse_config) == False):
            transition = algo.get_transition_from_parse_config(parse_config, example)
            if(transition == None):
                break
            sentence_instances.append((parse_config.get_state(), transition))
            """ update config based on transition if valid """
            parse_config = algo.update_state(parse_config, transition)
        training_instances += sentence_instances
    return training_instances

def is_final_state(parse_config):
    if parse_config.stack == [0] and parse_config.buffer == []:
        return True
    else:
        return False

if __name__ == '__main__':
    test_set = read_conll('./data/tel_dev.conll', lowercase=False)
    test_set = remove_non_projective_sentences(test_set)
    instances = get_training_instances(test_set)
    debug_file = open('test_configurations.txt', 'w')
    debug_file.write('predicted_label|' +'\t'+ 'stack|'+ '\t'+'buffer|'+'\t'+ 'dependencies|'+'\t' + 'sentence'+'\n')
    for instance in instances:
        line = [instance[1], str(instance[0]['stack']), str(instance[0]['buffer']), str(instance[0]['relations']), ' '.join(instance[0]['sentence'])]
        debug_file.write('|\t'.join(line) + '\n')

