
import os,sys
path= sys.argv[1]

with open(path, 'r') as f:
    lines = f.readlines()
sent1_list= []
sent2_list= []
for line in lines:
    line = line.rstrip("\n")
    try:
        if "||" in line:
            sent1, sent2 = line.split("||")
        else:
            sent1 = sent2 = line
    except:
        continue
    sent1_list.append(sent1)
    sent2_list.append(sent2)
print(sent2_list)