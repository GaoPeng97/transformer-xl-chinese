
from vocabulary import Vocab
import tensorflow as tf

'''
load model
tokenize
get log prob
next top_k
sample next
sample next iter
'''

input_text = "对于这威胁之话，萧炎却是没有半点理会，转过身来，徽眯的眼眸望向紫衣少女二人，声音之中，略有些惊异。"
tmp_Vocab = Vocab()
tmp_Vocab.count_file("../data/doupo/train.txt", add_eos=False)
tmp_Vocab.build_vocab()
# tokenlized_input = tmp_Vocab.encode_file("../data/doupo/sample.txt", ordered=True)
tokenlized_input = tmp_Vocab.encode_sents(input_text)

# saver = tf.train.Saver()
#
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     ckpt = tf.train.get_checkpoint_state("EXP-doupo")
#     saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
#     # saver.restore(sess, "EXP-doupo/model.ckpt")
