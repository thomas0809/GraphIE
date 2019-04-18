import os
import itertools
import sys
homedir = os.path.expanduser('~')

gpus_idx = list(map(int, sys.argv[1].split()))
gpu_idx = int(sys.argv[2])

group_idx = gpus_idx.index(gpu_idx)
num_groups = len(gpus_idx)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

hidden_sizes = [400]
tag_spaces = [128]
gammas = [0]
learning_rates = [0.01]
decay_rates = [0.025]

num_repeat = 1
# p_ems = [0.2, .3]
# p_ins = [.2, .3]
# p_rnns = [(0.2, 0.02)]
# p_outs = [0.2, .3]

# hidden_sizes = [256]
p_ems = [0.3]
p_ins = [.33]
p_rnns = [(0.3, 0.4)]
p_outs = [0.3]
batch_sizes = [16]

dropouts = ['std']
# dropouts = ['std']
char_methods = ['cnn']

parameters = [hidden_sizes, tag_spaces, gammas, learning_rates, decay_rates, p_ems, p_ins, p_rnns, p_outs, dropouts, char_methods, batch_sizes]
parameters = list(itertools.product(*parameters)) * num_repeat
parameters = list(split(parameters, num_groups))

dataset_name = 'WSJ-PTB'

params = parameters[group_idx]
for param in params:
	hidden_size, tag_space, gamma, learning_rate, decay_rate, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size = param
	result_file_path = 'results/hyperparameters_tuning_{}_{}'.format(dataset_name, gpu_idx)
	p_rnn = '{} {}'.format(p_rnn[0], p_rnn[1])
	log_msg = '\nhidden_size: {}\ttag_space: {}\tgamma: {}\tlearning_rate: {}\tdecay_rate: {}\tp_em: {}\tp_in: {}\tp_rnn: {}\tp_out: {}\ndropout: {}\tchar_method: {}\tbatch_size: {}\n'.format(hidden_size, tag_space, gamma, learning_rate, decay_rate, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size)
	with open(result_file_path, 'a') as ofile:
		ofile.write(log_msg)
	print(log_msg)
	command = 'CUDA_VISIBLE_DEVICES={} python examples/POSCRF_WSJ.py --cuda --mode LSTM --encoder_mode lstm --char_method {} --num_epochs 200 --batch_size {} --hidden_size {} --num_layers 1 \
				 --char_dim 30 --char_hidden_size 30 --tag_space {} --max_norm 10. --gpu_id {} --results_folder results --tmp_folder tmp --alphabets_folder data/alphabets \
				 --learning_rate {} --decay_rate {} --schedule 1 --gamma {} --o_tag O --dataset_name {} \
				 --dropout {} --p_em {} --p_in {} --p_rnn {} --p_out {} --unk_replace 0.0 --bigram --result_file_path {}\
				 --embedding glove --embedding_dict "/data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.100d.txt" \
				 --train "/data/medg/misc/jindi/nlp/dependency-stanford-3.3.0/PTB_train_gold.conll" --dev "/data/medg/misc/jindi/nlp/dependency-stanford-3.3.0/PTB_development_gold.conll" \
				 --test "/data/medg/misc/jindi/nlp/dependency-stanford-3.3.0/PTB_test_gold.conll"'.format(gpu_idx, char_method, batch_size, hidden_size, tag_space, gpu_idx, learning_rate, decay_rate, gamma, dataset_name, dropout, p_em, p_in, p_rnn, p_out, result_file_path)
	os.system(command)

