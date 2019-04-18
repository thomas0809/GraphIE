import os
import itertools
import sys
import argparse
from efficiency.log import show_time, del_quote

homedir = os.path.expanduser('~')

parser = argparse.ArgumentParser(
    description='Tuning with bi-directional RNN-CNN-CRF')
parser.add_argument(
    '--graph_model', choices=['gnn', 'transformer_graph'], help='architecture of rnn')
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

gpu_idx = args.gpu_id

hidden_sizes = [450]
num_repeat = 1

# this setting yields 91.26(91.26), 91.40 (91.64), 91.22 (91.37)
dropouts = ['gcn']
p_ems = [0.2]
p_ins = [.33]
p_rnns = [(0.33, 0.5, 0.5)]
p_tags = [0.5]

max_epochs = 400

learning_rate_gcns = [5e-4, 1e-3, 2e-4]
gcn_warmups = [200, 1000]
pretrain_lstms = [5]  # try 5
comb_mode=2

seed = 5

parameters = [hidden_sizes, learning_rate_gcns, gcn_warmups, pretrain_lstms,
              p_ems, p_ins, p_rnns, p_tags, dropouts]
parameters = list(itertools.product(*parameters)) * num_repeat

parameters = [(450, 0.001, 1000, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn')]

dataset_name = '03conll'
results_folder = './data/run/'

for param_i, param in enumerate(parameters):
    hidden_size, learning_rate_gcn, gcn_warmup, pretrain_lstm, \
    p_em, p_in, p_rnn, p_tag, dropout = param
    p_rnn = '{} {} {}'.format(p_rnn[0], p_rnn[1], p_rnn[2])

    misc = "{}".format(
        del_quote(str(param)))
    print("\n", misc, "\n")
    st_time = show_time()

    command = 'CUDA_VISIBLE_DEVICES={gpu_idx} python examples/NERCRF_conll.py ' \
              '--cuda --mode LSTM --encoder_mode lstm ' \
              '--char_method cnn --num_epochs {max_epochs} --batch_size 1 ' \
              '--hidden_size {hidden_size} --num_layers 1 ' \
              '--char_dim 30 --char_hidden_size 30 --tag_space 128 ' \
              '--max_norm 10. --gpu_id {gpu_idx} ' \
              '--alphabets_folder data/alphabets ' \
              '--learning_rate 0.01 --decay_rate 0.05 --schedule 1 ' \
              '--gamma 0. --o_tag O --dataset_name {dataset_name} ' \
              '--dropout {dropout} --p_em {p_em} --p_in {p_in} ' \
              '--p_rnn {p_rnn} --p_tag {p_tag} --unk_replace 0.0 ' \
              '--bigram ' \
              '--seed {seed} ' \
              '--learning_rate_gcn {learning_rate_gcn} --gcn_warmup {gcn_warmup} ' \
              '--pretrain_lstm {pretrain_lstm} ' \
              '--embedding glove --embedding_dict "data/pretr/emb/glove.6B.100d.txt" ' \
              '--train "data/dset/03co/train.c_w_d_dw_ds_sw_word_ibo_dic" ' \
              '--dev "data/dset/03co/valid.c_w_d_dw_ds_sw_word_ibo_dic" ' \
              '--test "data/dset/03co/test.c_w_d_dw_ds_sw_word_ibo_dic" ' \
              '--results_folder {results_folder} ' \
              '--uid {uid} --misc "{misc}" --smooth ' \
        .format(gpu_idx=gpu_idx, max_epochs=max_epochs, hidden_size=hidden_size,
                dataset_name=dataset_name, dropout=dropout,
                p_em=p_em, p_in=p_in, p_rnn=p_rnn, p_tag=p_tag,
                learning_rate_gcn=learning_rate_gcn, gcn_warmup=gcn_warmup,
                pretrain_lstm=pretrain_lstm,
                seed=seed, results_folder=results_folder,
                uid=st_time, misc=misc)

    os.system(command)
