
from vocabulary import Vocab
import tensorflow as tf
from absl import flags
'''
load model
tokenize
get log prob
next top_k
sample next
sample next iter
'''
#
# flags.DEFINE_integer("tgt_len", default=70,
#                      help="Number of steps to predict")
# flags.DEFINE_integer("mem_len", default=70,
#                      help="Number of steps to cache")
# flags.DEFINE_bool("same_length", default=False,
#                   help="Same length attention")
# flags.DEFINE_integer("clamp_len", default=-1,
#                      help="Clamp length")
#
# flags.DEFINE_integer("n_layer", default=6,
#                      help="Number of layers.")
# flags.DEFINE_integer("d_model", default=500,
#                      help="Dimension of the model.")
# flags.DEFINE_integer("d_embed", default=500,
#                      help="Dimension of the embeddings.")
# flags.DEFINE_integer("n_head", default=10,
#                      help="Number of attention heads.")
# flags.DEFINE_integer("d_head", default=50,
#                      help="Dimension of each attention head.")
# flags.DEFINE_integer("d_inner", default=1000,
#                      help="Dimension of inner hidden size in positionwise feed-forward.")
# flags.DEFINE_float("dropout", default=0.1,
#                    help="Dropout rate.")
# flags.DEFINE_float("dropatt", default=0.1,
#                    help="Attention dropout rate.")
# flags.DEFINE_bool("untie_r", default=False,
#                   help="untie r_w_bias and r_r_bias")
#
#
# def build_single_core_graph(n_token, cutoffs, inp, mems):
#         inp = tf.transpose(inp, [1, 0])
#         if FLAGS.init == "uniform":
#             initializer = tf.initializers.random_uniform(
#                 minval=-FLAGS.init_range,
#                 maxval=FLAGS.init_range,
#                 seed=None)
#         elif FLAGS.init == "normal":
#             initializer = tf.initializers.random_normal(
#                 stddev=FLAGS.init_std,
#                 seed=None)
#             proj_initializer = tf.initializers.random_normal(
#                 stddev=FLAGS.proj_init_std,
#                 seed=None)
#
#         tie_projs = [False for _ in range(len(cutoffs) + 1)]
#         if FLAGS.proj_share_all_but_first:
#             for i in range(1, len(tie_projs)):
#                 tie_projs[i] = True
#
#         output = model.transformer_inference(
#             dec_inp=inp,
#             mems=mems,
#             n_token=n_token,
#             n_layer=FLAGS.n_layer,
#             d_model=FLAGS.d_model,
#             d_embed=FLAGS.d_embed,
#             n_head=FLAGS.n_head,
#             d_head=FLAGS.d_head,
#             d_inner=FLAGS.d_inner,
#             dropout=FLAGS.dropout,
#             dropatt=FLAGS.dropatt,
#             initializer=initializer,
#             proj_initializer=proj_initializer,
#             is_training=False,
#             mem_len=FLAGS.mem_len,
#             cutoffs=cutoffs,
#             div_val=FLAGS.div_val,
#             input_perms=None,
#             same_length=FLAGS.same_length,
#             clamp_len=FLAGS.clamp_len,
#             use_tpu=False,
#             untie_r=FLAGS.untie_r,
#             proj_same_dim=True)
#
#         return output


input_text = "对于这威胁之话，萧炎却是没有半点理会，转过身来，徽眯的眼眸望向紫衣少女二人，声音之中，略有些惊异。"
tmp_Vocab = Vocab()
tmp_Vocab.count_file("../data/doupo/train.txt", add_eos=False)
tmp_Vocab.build_vocab()
# tokenlized_input = tmp_Vocab.encode_file("../data/doupo/sample.txt", ordered=True)
encoded_input = tmp_Vocab.encode_sents(input_text)

# saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    new_saver = tf.train.import_meta_graph('EXP-doupo3/model-0.00020474523265875177.ckpt.meta')
    # new_saver.restore('')
    graph = tf.get_default_graph()
    # x = graph.get_operation_by_name('')
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    print(tensor_name_list)


#
# def main(unused_argv):
#     del unused_argv  # Unused
#
#     tf.logging.set_verbosity(tf.logging.INFO)
#
#     # Get corpus info
#     corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
#     n_token = corpus_info["vocab_size"]
#     cutoffs = corpus_info["cutoffs"][1:-1]
#     tf.logging.info("n_token {}".format(n_token))
#
#     if FLAGS.do_train:
#         train(n_token, cutoffs, "/gpu:0")
#     if FLAGS.do_eval:
#         evaluate(n_token, cutoffs, "/gpu:0")
#
#
# # new added by pgao
# def inference(n_token, cutoffs, ps_device):
#     # Get input function and model function
#     eval_input_fn, eval_record_info = data_utils.get_input_fn(
#         record_info_dir=FLAGS.record_info_dir,
#         split=FLAGS.eval_split,
#         per_host_bsz=FLAGS.eval_batch_size,
#         tgt_len=FLAGS.tgt_len,
#         num_core_per_host=FLAGS.num_core_per_host,
#         num_hosts=1,
#         use_tpu=False)
#
#     num_batch = eval_record_info["num_batch"]
#     if FLAGS.max_eval_batch > 0:
#         num_batch = FLAGS.max_eval_batch
#     tf.logging.info("num of batches {}".format(num_batch))
#
#     # Create computational graph
#     eval_set = eval_input_fn({
#         "batch_size": FLAGS.eval_batch_size,
#         "data_dir": FLAGS.data_dir})
#
#     input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()
#
#     inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
#     labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)
#
#     per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
#     tower_mems, tower_losses, tower_new_mems = [], [], []
#
#     for i in range(FLAGS.num_core_per_host):
#         with tf.device(assign_to_gpu(i, ps_device)), \
#              tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
#             mems_i = [tf.placeholder(tf.float32,
#                                      [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
#                       for _ in range(FLAGS.n_layer)]
#
#             loss_i, new_mems_i = single_core_graph(
#                 n_token=n_token,
#                 cutoffs=cutoffs,
#                 is_training=False,
#                 inp=inputs[i],
#                 tgt=labels[i],
#                 mems=mems_i)
#
#             tower_mems.append(mems_i)
#             tower_losses.append(loss_i)
#             tower_new_mems.append(new_mems_i)
#
#     # sum losses across towers
#     if len(tower_losses) > 1:
#         loss = tf.add_n(tower_losses) / len(tower_losses)
#     else:
#         loss = tower_losses[0]
#
#     # Evaluation loop
#     tower_mems_np = [
#         [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
#          for layer in range(FLAGS.n_layer)]
#         for core in range(FLAGS.num_core_per_host)
#     ]
#
#     saver = tf.train.Saver()
#
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         sess.run(tf.global_variables_initializer())
#
#         if FLAGS.eval_ckpt_path is None:
#             eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
#         else:
#             eval_ckpt_path = FLAGS.eval_ckpt_path
#         tf.logging.info("Evaluate {}".format(eval_ckpt_path))
#         saver.restore(sess, eval_ckpt_path)
#
#         fetches = [loss, tower_new_mems, tf.size(label_feed)]
#
#         format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
#             len(str(num_batch)))
#
#         total_loss, total_cnt = 0, 0
#         for step in range(num_batch):
#             if step % (num_batch // 10) == 0:
#                 tf.logging.info(format_str.format(step, num_batch))
#
#             feed_dict = {}
#             for i in range(FLAGS.num_core_per_host):
#                 for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
#                     feed_dict[m] = m_np
#
#             fetched = sess.run(fetches, feed_dict=feed_dict)
#
#             loss_np, tower_mems_np, cnt_np = fetched[:3]
#             total_loss += loss_np * cnt_np
#             total_cnt += cnt_np
#
#         avg_loss = total_loss / total_cnt
#         tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
#             avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))
#
#
#
#
# if __name__ == "__main__":
#     tf.app.run()
#
#


