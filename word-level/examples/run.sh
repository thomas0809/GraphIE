CUDA_VISIBLE_DEVICES=0 python -m pdb -c continue examples/NERCRF_conll.py --cuda --mode LSTM --encoder_mode lstm --char_method cnn --num_epochs 200 --batch_size 16 --hidden_size 400 --num_layers 1 \
--char_dim 30 --char_hidden_size 30 --tag_space 128 --max_norm 10. --gpu_id 0 --results_folder results --tmp_folder tmp --alphabets_folder data/alphabets \
--learning_rate 0.01 --decay_rate 0.05 --schedule 1 --gamma 0. --evaluate_raw_format --o_tag Other --dataset_name conll \
--dropout weight_drop --p_em 0.33 --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --unk_replace 0.0 --bigram --result_file_path "results/hyperparameters_tuning" \
--embedding glove --embedding_dict "../../data/pretrained/glove/glove.6B.100d.txt" \
--train "data/conll2003/eng.train.bio.conll.simple" --dev "data/conll2003/eng.dev.bio.conll.simple" \
--test "data/conll2003/eng.test.bio.conll.simple"

# std?

CUDA_VISIBLE_DEVICES=1 python -m pdb -c continue examples/NERCRF_conll.py --cuda --mode LSTM --encoder_mode lstm --char_method cnn --num_epochs 200 --batch_size 16 --hidden_size 400 --num_layers 1 \
--char_dim 30 --char_hidden_size 30 --tag_space 128 --max_norm 10. --gpu_id 1 --results_folder results --tmp_folder tmp --alphabets_folder data/alphabets \
--learning_rate 0.01 --decay_rate 0.05 --schedule 1 --gamma 0. --evaluate_raw_format --o_tag Other --dataset_name conll \
--dropout std --p_em 0.33 --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --unk_replace 0.0 --bigram --result_file_path "results/hyperparameters_tuning" \
--embedding glove --embedding_dict "../../data/pretrained/glove/glove.6B.100d.txt" \
--train "data/conll2003/eng.train.bio.conll.simple" --dev "data/conll2003/eng.dev.bio.conll.simple" \
--test "data/conll2003/eng.test.bio.conll.simple" \
