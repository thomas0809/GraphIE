import os
import itertools
import sys
homedir = os.path.expanduser('~')
import json

cluster_file = sys.argv[1]
gpu_idx = int(sys.argv[2])
eval_dataset_type = sys.argv[3]

hidden_size = 450
dropout = 'weight_drop' # this setting got results: 91.77(91.84), 91.39(91.47), 91.47(91.47)
p_em = 0.2
p_in = .3
p_rnn = (0.2, 0.02) # 0.2 should be better than 0.3
p_out = 0.2 # 0.2 is mostly better

# dropouts = ['std'] # this setting yields 91.26(91.26), 91.40 (91.64), 91.22 (91.37)
# p_ems = [0.2]
# p_ins = [.33]
# p_rnns = [(0.33, 0.5)]
# p_outs = [0.5]

batch_size = 16
char_method = 'cnn'
dataset_name = 'conll'

result_file_path = 'results/cluster_performance_{}_{}_{}'.format(dataset_name, eval_dataset_type, gpu_idx)
# save_checkpoint = 'results/hidden_size-{}_dropout-{}_p_em-{}_p_in-{}_p_rnn-{}_p_out-{}_repeat_time-{}'.format(hidden_size, dropout, p_em, p_in, p_rnn, p_out, idx)
save_checkpoint = 'results/network_conll'
p_rnn = '{} {}'.format(p_rnn[0], p_rnn[1])

# read the original train file
all_sentences = dict()
with open('data/conll2003/eng.{}.bio.conll.simple.indexed'.format(eval_dataset_type), 'r') as ifile:
	sentence = ''
	for line in ifile:
		if line.startswith('###'):
			if sentence:
				all_sentences[line_idx] = sentence
				sentence = ''
			line_idx = int(line.strip().split('#')[-1])
		else:
			sentence += line

# read the cluster index file
cluster_dict = json.load(open(cluster_file, 'r'))[eval_dataset_type]
for cluster_idx, cluster_sent_idx in cluster_dict.items():
	evaluation_data_tmp = 'data/conll2003/eng.{}.bio.conll.simple.cluster.{}'.format(eval_dataset_type, cluster_idx)
	eval_filename = 'tmp/{}_conll_cluster_{}'.format(eval_dataset_type, cluster_idx)
	with open(evaluation_data_tmp, 'w') as ofile:
		for sent_idx in cluster_sent_idx:
			ofile.write(all_sentences[int(sent_idx)+1]+'\n')

	log_msg = 'Cluster {} (data size: {}):\n'.format(cluster_idx, len(cluster_sent_idx))
	with open(result_file_path, 'a') as ofile:
		ofile.write(log_msg)
	print(log_msg)
	command = 'CUDA_VISIBLE_DEVICES={} python examples/NERCRF_evaluation.py --cuda --mode LSTM --encoder_mode lstm --char_method {} --num_epochs 200 --batch_size {} --hidden_size {} --num_layers 1 \
				 --char_dim 30 --char_hidden_size 30 --tag_space 128 --max_norm 10. --gpu_id {} --results_folder results --tmp_folder tmp --alphabets_folder data/alphabets \
				 --learning_rate 0.01 --decay_rate 0.05 --schedule 1 --gamma 0. --o_tag O --dataset_name conll --save_checkpoint {} \
				 --dropout {} --p_em {} --p_in {} --p_rnn {} --p_out {} --unk_replace 0.0 --bigram --result_file_path {} --eval_filename {} --restore \
				 --embedding glove --embedding_dict "/data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.100d.txt" \
				 --train "data/conll2003/eng.train.bio.conll.simple" --dev "data/conll2003/eng.dev.bio.conll.simple" \
				 --test {}'.format(gpu_idx, char_method, batch_size, hidden_size, gpu_idx, save_checkpoint, dropout, p_em, p_in, p_rnn, p_out, result_file_path, eval_filename, evaluation_data_tmp)
	os.system(command)
with open(result_file_path, 'a') as ofile:
	ofile.write('\n\n')

