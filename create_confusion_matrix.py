import sys
from collections import defaultdict
gold_f = open(sys.argv[1], 'r')
sys_f = open(sys.argv[2], 'r')

gold_labels = []

for line in gold_f.readlines():
    parts = line.split('\t')
    if len(parts)==10:
        gold_labels.append(parts[7])

sys_labels = []
for line in sys_f.readlines():
    parts = line.split('\t')
    if len(parts)==10:
        sys_labels.append(parts[7])

label_set = set(sorted(gold_labels))
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

