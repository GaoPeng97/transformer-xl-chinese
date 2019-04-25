
from vocabulary import Vocab
import tensorflow as tf
from gpu_utils import assign_to_gpu
from absl import flags
'''
load model
tokenize
get log prob
next top_k
sample next
sample next iter
'''


def parser(record):
    # preprocess "inp_perm" and "tgt_perm"
    record_spec = {
        "inputs": tf.VarLenFeature(tf.int64),
    }

    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)

    for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
            val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
            val = tf.to_int32(val)
        example[key] = val

    return example["inputs"]


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


# input_text = "测试测试测试测试"
input_text = "对于这威胁之话，萧炎却徽眯的眼眸望向紫衣少女二人，声音之中，略有些惊异。"
tmp_Vocab = Vocab()
tmp_Vocab.count_file("../data/doupo/train.txt", add_eos=False)
tmp_Vocab.build_vocab()
# encoded_input = tmp_Vocab.encode_file("../data/doupo/sample.txt", ordered=True)
encoded_input = tmp_Vocab.encode_sents(input_text, ordered=True)

feature = {
    "inputs": _int64_feature(encoded_input)
}

save_path = '../data/doupo/tmp.tfrecords'
record_writer = tf.python_io.TFRecordWriter(save_path)
example = tf.train.Example(features=tf.train.Features(feature=feature))
record_writer.write(example.SerializeToString())


test_list = tf.placeholder(tf.int64, shape=[1])
dataset = tf.data.Dataset.from_tensors(test_list)
# dataset = tf.data.TFRecordDataset(dataset)
# dataset = dataset.map(parser)
dataset = dataset.batch(1, drop_remainder=True)


# input_feed = dataset.make_one_shot_iterator().get_next()
iterator = dataset.make_initializable_iterator()
input_feed = iterator.get_next()
# inputs = tf.split(input_feed, 1, 0)

feed_dict = {test_list : [1]}

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict=feed_dict)
    for i in range(1):
        value = sess.run(input_feed)
        print(value)
        print('========================== end ===============================')


#
# # saver = tf.train.Saver()
#
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#
#     new_saver = tf.train.import_meta_graph('EXP-doupo3/model-0.00020474523265875177.ckpt.meta')
#     # new_saver.restore('')
#     graph = tf.get_default_graph()
#     # x = graph.get_operation_by_name('')
#     tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
#     print(tensor_name_list)

