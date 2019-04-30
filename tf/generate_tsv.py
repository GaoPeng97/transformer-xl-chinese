from vocabulary import Vocab
import csv

tmp_Vocab = Vocab()
tmp_Vocab.count_file("../data/test/train.txt", add_eos=False)
tmp_Vocab.build_vocab()

with open('../data/test/label.tsv', 'wt') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(['label', 'index'])
    for i in range(len(tmp_Vocab.idx2sym)):
        tsv_writer.writerow([tmp_Vocab.idx2sym[i], i])
        # tsv_writer.writerow([tmp_Vocab.idx2sym[i]])
