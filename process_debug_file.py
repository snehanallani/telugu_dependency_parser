import sys
from collections import defaultdict

g_name = sys.argv[1]
s_name = sys.argv[2]
gold = open(g_name, 'r')
sys = open(s_name, 'r')

gold_out = defaultdict(list)
sys_out = defaultdict(list)

for line in gold.readlines():
    parts = line.split('|\t')
    gold_out[parts[4]].append(parts[:4])

for line in sys.readlines():
    parts = line.split('|\t')
    sys_out[parts[4]].append(parts[:4])

g_config = open('gold_configurations', 'w')
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
    
print('Matching sentences: ', counter)
