import os
import itertools
import sys
import argparse
from efficiency.log import show_time
homedir = os.path.expanduser('~')

parser = argparse.ArgumentParser(
    description='Tuning with bi-directional RNN-CNN-CRF')
parser.add_argument(
    '--graph_model', choices=['gnn', 'transformer_graph'], help='architecture of rnn')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--char_dim', type=int, default=25)
parser.add_argument('--char_hidden_size', type=int, default=25)

args = parser.parse_args()

gpu_idx = args.gpu_id


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


hidden_sizes = [200] # now 200 is the best, much better than 250 and 150
num_repeat = 1

# dropouts = ['weight_drop'] # this setting got results: 91.77(91.84), 91.39(91.47), 91.47(91.47)
# p_ems = [0.2]
# p_ins = [.3]
# p_rnns = [(0.2, 0.02)] # 0.2 should be better than 0.3
# p_outs = [0.2] # 0.2 is mostly better

# this setting yields 91.26 (91.26), 91.40 (91.64), 91.22 (91.37)
dropouts = ['std']
p_ems = [0.1]
p_ins = [.33]
p_rnns = [(0.33, 0.5)]
p_outs = [0.4] # 0.4 is a little better

char_dims = [40] # 200 should be better than 150
char_hidden_sizes = [40]
tag_spaces = [64] # 64 is much better than 96

batch_sizes = [32]
char_methods = ['cnn']

parameters = [hidden_sizes, p_ems, p_ins, p_rnns,
              p_outs, dropouts, char_methods, batch_sizes,
              char_dims, char_hidden_sizes, tag_spaces]

parameters = list(itertools.product(*parameters)) * num_repeat
import pdb; pdb.set_trace()
dataset_name = 'chemdner'

for param in parameters:
    hidden_size, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size, \
    char_dim, char_hidden_size, tag_space = param

    st_time = show_time(cat_server=True)

    result_file_path = '/afs/csail.mit.edu/u/z/zhijing/proj/ie/data/run/az/hyperp_{}_{}'.format(
        dataset_name, st_time)
    p_rnn = '{} {}'.format(p_rnn[0], p_rnn[1])

    log_msg = '\n{}, char_dim: {}, char_hidden_size: {}'.format(
        st_time, char_dim, char_dim)
    log_msg += '\nhidden_size: {}\tp_em: {}\tp_in: {}\tp_rnn: {}\tp_out: {}\tdropout: {}\tchar_method: {}\tbatch_size: {}\n'.format(
        hidden_size, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size)

    print(log_msg)
    command = 'CUDA_VISIBLE_DEVICES={} python -m pdb -c continue examples/NERCRF_conll.py --cuda --mode LSTM --encoder_mode lstm --char_method {} --num_epochs 150 --batch_size {} --hidden_size {} --num_layers 1 \
				 --char_dim {} --char_hidden_size {} --tag_space 64 --max_norm 15. --gpu_id {} --results_folder results --tmp_folder tmp --alphabets_folder data/chem/alphabets \
				 --learning_rate 0.005 --decay_rate 0.01 --schedule 1 --gamma 0. --o_tag O --dataset_name {} \
				 --dropout {} --p_em {} --p_in {} --p_rnn {} --p_out {} --unk_replace 0.0 --bigram --result_file_path {} \
				 --embedding glove --embedding_dict "/afs/csail.mit.edu/u/z/zhijing/proj/data/pretr/emb/chemdner_pubmed_drug.word2vec_model_token4_d50" \
				 --train "../../../data/dsets/chem/dalian/train.c_w_d_dw_word_ibo_dic" \
				 --dev "../../../data/dsets/chem/dalian/dev.c_w_d_dw_word_ibo_dic" \
				 --test "../../../data/dsets/chem/doc_level/evaluation.c_w_d_dw_word_ibo_dic" \
                 --uid {}'\
                .format(gpu_idx, char_method, batch_size, hidden_size, char_dim, char_dim, gpu_idx,
                dataset_name, dropout, p_em, p_in, p_rnn, p_out, result_file_path, st_time)
    os.system(command)

