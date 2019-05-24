import os
import itertools
import sys
homedir = os.path.expanduser('~')

gpus_idx = map(int, sys.argv[1].split())
gpu_idx = int(sys.argv[2])

group_idx = gpus_idx.index(gpu_idx)
num_groups = len(gpus_idx)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

hidden_sizes = [400]
num_repeat = 3
# p_ems = [0.2, .3]
# p_ins = [.2, .3]
# p_rnns = [(0.2, 0.02)]
# p_outs = [0.2, .3]

# hidden_sizes = [256]
p_ems = [0.33]
p_ins = [.33]
p_rnns = [(0.33, 0.5)]
p_outs = [0.5]
batch_sizes = [80]

dropouts = ['std']
# dropouts = ['std']
char_methods = ['cnn']

parameters = [hidden_sizes, p_ems, p_ins, p_rnns, p_outs, dropouts, char_methods, batch_sizes]
parameters = list(itertools.product(*parameters)) * num_repeat
parameters = list(split(parameters, num_groups))

dataset_name = 'conll'

params = parameters[group_idx]
for param in params:
	hidden_size, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size = param
	result_file_path = 'results/hyperparameters_tuning_large_batch_size_{}_{}'.format(dataset_name, gpu_idx)
	p_rnn = '{} {}'.format(p_rnn[0], p_rnn[1])
	log_msg = '\nhidden_size: {}\tp_em: {}\tp_in: {}\tp_rnn: {}\tp_out: {}\tdropout: {}\tchar_method: {}\tbatch_size: {}\n'.format(hidden_size, p_em, p_in, p_rnn, p_out, dropout, char_method, batch_size)
	with open(result_file_path, 'a') as ofile:
		ofile.write(log_msg)
	print(log_msg)
	command = 'CUDA_VISIBLE_DEVICES={} python examples/NERCRF_conll_large_batch_size.py --cuda --mode LSTM --encoder_mode lstm --char_method {} --num_epochs 200 --batch_size {} --hidden_size {} --num_layers 1 \
				 --char_dim 30 --char_hidden_size 30 --tag_space 128 --max_norm 10. --gpu_id {} --results_folder results --tmp_folder tmp --alphabets_folder data/alphabets \
				 --learning_rate 0.01 --decay_rate 0.03 --schedule 1 --gamma 0. --evaluate_raw_format --o_tag Other --dataset_name conll \
				 --dropout {} --p_em {} --p_in {} --p_rnn {} --p_out {} --unk_replace 0.0 --bigram --result_file_path {}\
				 --embedding glove --embedding_dict "/data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.100d.txt" \
				 --train "data/conll2003/eng.train.bio.conll.simple" --dev "data/conll2003/eng.dev.bio.conll.simple" \
				 --test "data/conll2003/eng.test.bio.conll.simple"'.format(gpu_idx, char_method, batch_size, hidden_size, gpu_idx, dropout, p_em, p_in, p_rnn, p_out, result_file_path)
	os.system(command)

