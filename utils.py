import numpy as np

def cmp_result(label,rec):
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def load_dict(dict_file):
    with open(dict_file) as dict_f:
        dict_data = dict_f.readlines()

    lexicon={}
    num_line = 0
    for line in dict_data:
        num_line += 1
        word=line.strip().split()
        lexicon[word[0]]=int(word[1])
    print('total words/phones',len(lexicon))
    return lexicon, num_line
