import sys
from collections import defaultdict
gold_out = open('./gold_out.conll', 'r')
sys_out = open('./sys_out.conll', 'r')

gold_instances = open('./gold_instances.txt', 'r')
sys_instances = open('./sys_instances.txt', 'r')

gold_label = sys.argv[1]
sys_label = sys.argv[2]

gold_data = defaultdict(dict) # {sentence:{'labels':[], 'configurations':[]}}
sys_data = defaultdict(dict)

def populate_label_data(out_lines, data_dict):
    all_labels = []
    words = []
    labels = []
    for line in out_lines:
        parts = line.split('\t')
        if len(parts)==10:
            words.append(parts[1])
            labels.append(parts[7])
        elif line.strip()=='':
            all_labels += labels
            data_dict[' '.join(words)] = {'labels':labels, 'configurations':[]}
            words = []
            labels = []
    return all_labels


gold_lines = gold_out.readlines()
gold_labels = populate_label_data(gold_lines, gold_data)
sys_labels = populate_label_data(sys_out.readlines(), sys_data)

for line in gold_instances.readlines()[1:]:
    parts = line.split('|\t')
    words = parts[4].split(' ')
    sent = ' '.join(words[1:]).strip()
    gold_data[sent]['configurations'].append(parts[:4])

for line in sys_instances.readlines()[1:]:
    parts = line.split('|\t')
    words = parts[4].split(' ')
    sent = ' '.join(words[1:]).strip()
    sys_data[sent]['configurations'].append(parts[:4])

f = open(gold_label + '_' + sys_label + '_mismatch.txt','w')

counter = 0
for sent, v in gold_data.items():
    gold_labels = v['labels']
    sys_labels = sys_data[sent]['labels']
    for i, label in enumerate(gold_labels):
        if not label == sys_labels[i]:
            if label == gold_label and sys_labels[i] == sys_label:
                counter += 1
                f.write(sent +'\n')
                f.write('Gold Labels : ' + str(gold_labels) + '\n')
                f.write('System Labels : ' + str(sys_labels) + '\n\n')
                f.write('Gold_Configurations : \n')
                for c in v['configurations']:
                    f.write(str(c) + '\n')
                f.write('\nSystem_Configurations : \n')
                for c in sys_data[sent]['configurations']:
                    f.write(str(c) + '\n')
                f.write(80*'-'+'\n\n')

print(gold_label, ' , ', sys_label, ' mismatched sentences ', counter)

"""g_config = open('gold_configurations', 'w')
s_config = open('sys_configurations', 'w')

counter = 0

for key in sorted(gold_out):
    if gold_out[key] == sys_out[key] :
        counter += 1
    else:
        g_config.write(key)
        s_config.write(key)
        for v in gold_out[key]:
            g_config.write(2*'\t' + '\t\t'.join(str(i) for i in v) + '\n')
        for v in sys_out[key]:
            s_config.write(2*'\t' + '\t\t'.join(str(i) for i in v) + '\n')
    
print('Matching sentences: ', counter)"""

"""label_set = set(sorted(gold_labels))
label_to_ix = { i : label for i, label in enumerate(label_set)}

confusion = defaultdict(dict)
g_counter = defaultdict(int)

assert len(sys_labels) == len(gold_labels)
for idx, label in enumerate(gold_labels):
    sys_label = sys_labels[idx]
    g_counter[label] += 1
    if sys_label in confusion[label].keys():
        confusion[label][sys_label] += 1
    else:
        confusion[label][sys_label] = 1

for key, val in sorted(g_counter.items(), key=lambda x: x[1], reverse=True):
    print('(', key,',', val, ')', ':\t', sorted(confusion[key].items(), key=lambda item: item[1], reverse=True))



gold_out = defaultdict(list)
sys_out = defaultdict(list)"""

