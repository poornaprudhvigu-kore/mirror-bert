CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--model_dir "roberta-base" \
	--train_dir "./data/mirror_corpus/stsbenchmark_train_10k_sents_random_sampled.txt" \
	--output_dir tmp/roberta_base_mirror_stsbenchmark_train_10k_sent_infonce0.04_maxlen50_bs200_mask5_dropout0.1_drophead0.0_cls \
	--epoch 1 \
	--train_batch_size 200 \
	--learning_rate 2e-5 \
	--max_length 50 \
	--infoNCE_tau 0.04 \
	--dropout_rate 0.1 \
	--drophead_rate 0.0 \
	--random_span_mask 5 \
	--agg_mode "cls" \
	--amp \
	--parallel \
	--use_cuda \
 	--save_checkpoint_all

