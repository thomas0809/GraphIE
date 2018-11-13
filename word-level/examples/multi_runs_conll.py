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
parser.add_argument('--expe_id', type=int, default=0)


args = parser.parse_args()

gpu_idx = args.gpu_id


hidden_sizes = [450]
num_repeat = 1

# dropouts = ['weight_drop'] # this setting got results: 91.77(91.84), 91.39(91.47), 91.47(91.47)
# p_ems = [0.2]
# p_ins = [.3]
# p_rnns = [(0.2, 0.02)] # 0.2 should be better than 0.3
# p_outs = [0.2] # 0.2 is mostly better

dropouts = ['gcn']  # this setting yields 91.26(91.26), 91.40 (91.64), 91.22 (91.37)
p_ems = [0.2]
p_ins = [.33]
p_rnns = [(0.33, 0.5, 0.5)]
p_tags = [0.5]

lambda1= 1
lambda2 = 0

max_epochs = 400

learning_rate_gcns = [5e-4, 1e-3, 2e-4]
gcn_warmups = [200, 1000]
pretrain_lstms = [5] # try 5
adj_attn = ''

seed = 1


parameters = [hidden_sizes, learning_rate_gcns, gcn_warmups, pretrain_lstms,
              p_ems, p_ins, p_rnns, p_tags, dropouts]
parameters = list(itertools.product(*parameters)) * num_repeat

parameters = [(450, 0.0005, 200, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0005, 1000, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.001, 200, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.001, 1000, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0002, 200, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0002, 1000, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0005, 200, 10, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0005, 200, 30, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0005, 1000, 10, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0005, 1000, 30, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.001, 200, 10, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.001, 200, 30, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.001, 1000, 10, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.001, 1000, 30, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0002, 200, 10, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0002, 200, 30, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0002, 1000, 10, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'), (450, 0.0002, 1000, 30, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn')]
# parameters = [(450, 0.001, 1000, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn')]
# parameters = [(450, 0.001, 200, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn'),
#               (450, 0.0005, 1000, 10, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn')]
parameters = [(450, 0.001, 1000, 5, 0.2, 0.33, (0.33, 0.5, 0.5), 0.5, 'gcn')]
parameters_subset = parameters[args.expe_id*2 : args.expe_id*2 + 2 ]

dataset_name = '03conll'

for param_i, param in enumerate(parameters_subset):

    hidden_size, learning_rate_gcn, gcn_warmup, pretrain_lstm, \
        p_em, p_in, p_rnn, p_tag, dropout = param
    st_time = show_time(cat_server=True)

    misc = "adj_attn={}, comb_mode=2, lambda:{},{}, seed:{}, {}".format(
        parameters.index(param),
        adj_attn, lambda1, lambda2, seed, del_quote(str(param)))
    misc = "adj_attn={}, comb_mode=2, lambda:{},{}, seed:{}, {}".format(

        adj_attn, lambda1, lambda2, seed, del_quote(str(param)))

    result_file_path = '/afs/csail.mit.edu/u/z/zhijing/proj/ie/data/run/az/hyperp_{}_{}'.format(
        dataset_name, st_time)
    p_rnn = '{} {} {}'.format(p_rnn[0], p_rnn[1], p_rnn[2])

    print("\n", misc, "\n")


    command = 'CUDA_VISIBLE_DEVICES={gpu_idx} python examples/NERCRF_conll.py --cuda --mode LSTM --encoder_mode lstm \
                --char_method cnn --num_epochs {max_epochs} --batch_size 1 --hidden_size {hidden_size} --num_layers 1 \
				 --char_dim 30 --char_hidden_size 30 --tag_space 128 --max_norm 10. --gpu_id {gpu_idx} --results_folder results \
				 --tmp_folder tmp --alphabets_folder data/alphabets \
				 --learning_rate 0.01 --decay_rate 0.05 --schedule 1 --gamma 0. --o_tag O --dataset_name {dataset_name} \
				 --dropout {dropout} --p_em {p_em} --p_in {p_in} --p_rnn {p_rnn} --p_tag {p_tag} --unk_replace 0.0 --bigram --result_file_path {result_file_path} \
				 --lambda1 {lambda1} --lambda2 {lambda2} \
                 --seed {seed} \
				 --learning_rate_gcn {learning_rate_gcn} --gcn_warmup {gcn_warmup} --pretrain_lstm {pretrain_lstm} \
				 --embedding glove --embedding_dict "../../../data/pretr/emb/glove.6B.100d.txt" \
				 --train "../../../data/dsets/03co/doc_level/train.c_w_d_dw_ds_sw_word_ibo_dic" \
				 --dev "../../../data/dsets/03co/doc_level/valid.c_w_d_dw_ds_sw_word_ibo_dic" \
				 --test "../../../data/dsets/03co/doc_level/test.c_w_d_dw_ds_sw_word_ibo_dic" \
                 --uid {uid} --misc "{misc}" --smooth' \
        .format(gpu_idx=gpu_idx, max_epochs=max_epochs, hidden_size=hidden_size,
                dataset_name=dataset_name, dropout=dropout, p_em=p_em, p_in=p_in, p_rnn=p_rnn, p_tag=p_tag,
                adj_attn=adj_attn, lambda1=lambda1, lambda2=lambda2, learning_rate_gcn=learning_rate_gcn , gcn_warmup=gcn_warmup, pretrain_lstm=pretrain_lstm,
                seed=seed, result_file_path=result_file_path,
                uid=st_time, misc=misc)

    os.system(command)

