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

hidden_sizes = [500]
num_repeat = 3

# dropouts = ['weight_drop'] # this setting got results: 91.77(91.84), 91.39(91.47), 91.47(91.47)
# p_ems = [0.2]
# p_ins = [.3]
# p_rnns = [(0.2, 0.02)] # 0.2 should be better than 0.3
# p_outs = [0.2] # 0.2 is mostly better

dropouts = ['std'] # this setting yields 91.26(91.26), 91.40 (91.64), 91.22 (91.37)
p_ems = [0.2]
p_ins = [.2]
p_rnns = [(0.2, 0.5)]
p_outs = [0.3]

batch_sizes = [32]
char_methods = ['cnn']

parameters = [hidden_sizes, p_ems, p_ins, p_rnns, p_outs, dropouts, char_methods, batch_sizes]
parameters = list(itertools.product(*parameters)) * num_repeat
parameters = list(split(parameters, num_groups))

dataset_name = 'ontonotes'

params = parameters[group_idx]
for idx, param in enumerate(params):
	hidden_size, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size = param
	result_file_path = 'results/hyperparameters_tuning_{}_{}'.format(dataset_name, gpu_idx)
	# save_checkpoint = 'results/hidden_size-{}_dropout-{}_p_em-{}_p_in-{}_p_rnn-{}_p_out-{}_repeat_time-{}'.format(hidden_size, dropout, p_em, p_in, p_rnn, p_out, idx)
	save_checkpoint = 'results/network_ontonotes'
	p_rnn = '{} {}'.format(p_rnn[0], p_rnn[1])
	log_msg = '\nhidden_size: {}\tp_em: {}\tp_in: {}\tp_rnn: {}\tp_out: {}\tdropout: {}\tchar_method: {}\tbatch_size: {}\n'.format(hidden_size, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size)
	with open(result_file_path, 'a') as ofile:
		ofile.write(log_msg)
	print(log_msg)
	command = 'CUDA_VISIBLE_DEVICES={} python examples/NERCRF_conll.py --cuda --mode LSTM --encoder_mode lstm --char_method {} --num_epochs 200 --batch_size {} --hidden_size {} --num_layers 1 \
				 --char_dim 30 --char_hidden_size 30 --tag_space 128 --max_norm 10. --gpu_id {} --results_folder results --tmp_folder tmp --alphabets_folder data/alphabets \
				 --learning_rate 0.01 --decay_rate 0.05 --schedule 1 --gamma 0. --o_tag O --dataset_name {} --save_checkpoint {} \
				 --dropout {} --p_em {} --p_in {} --p_rnn {} --p_out {} --unk_replace 0.0 --bigram --result_file_path {}\
				 --embedding glove --embedding_dict "/data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.100d.txt" \
				 --train "/data/medg/misc/jindi/nlp/ontonotes_doc_level/ontonotes.train.iob.indexed" --dev "/data/medg/misc/jindi/nlp/ontonotes_doc_level/ontonotes.development.iob.indexed" \
				 --test "/data/medg/misc/jindi/nlp/ontonotes_doc_level/ontonotes.test.iob.indexed"'.format(gpu_idx, char_method, batch_size, hidden_size, gpu_idx, dataset_name, save_checkpoint, dropout, p_em, p_in, p_rnn, p_out, result_file_path)
	os.system(command)

