import numpy as np
import random


def top_one_result(tmp_list):
    index_list = sorted(range(len(tmp_list)), key=lambda k: tmp_list[k], reverse=True)[:1]
    return index_list[0]


def gen_on_keyword(tmp_Vocab, keyword, tmp_list, lookup_table):

    keyword_index = tmp_Vocab.get_idx(keyword)
    index_list = sorted(range(len(tmp_list)), key=lambda k: tmp_list[k], reverse=True)[:3]

    if (float(tmp_list[index_list[0]]) / tmp_list[index_list[1]] > 1.3):
        return index_list[0]
    # keyword_index_2 = tmp_Vocab.get_idx('震')

    # print(np.sum(np.array(lookup_table[keyword_index]) * np.array(lookup_table[keyword_index_2])))

    # similar = 0
    index = 0
    for i in range(len(index_list)):
        if (i == 0):
            # similar = abs(np.sum(np.array(lookup_table[keyword_index]) * np.array(lookup_table[index_list[0]])))
            similar = np.linalg.norm(np.array(lookup_table[keyword_index]) - np.array(lookup_table[index_list[0]]))
        else:
            dist = np.linalg.norm(np.array(lookup_table[keyword_index]) - np.array(lookup_table[index_list[i]]))
            if (dist < similar):
                similar = dist
                index = i

    return index_list[index]


def gen_diversity(tmp_list):
    index_list = sorted(range(len(tmp_list)), key=lambda k: tmp_list[k], reverse=True)[:2]
    index = random.sample(index_list, 1)[0]
    if (float(tmp_list[index_list[0]]) / tmp_list[index_list[1]] > 1.3):
        index = index_list[0]
    return index
