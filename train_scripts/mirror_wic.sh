infoce=0.04
maxlen=50
bs=200
dropout=0.4
agg='tokenmarker4layer'
rs=33
data=$2
model="bert-base-uncased"
CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--model_dir ${model} \
	--train_dir "${data}"  \
	--output_dir ../tmp/${model}_mirror	\
	--use_cuda \
	--epoch 2 \
	--train_batch_size ${bs} \
	--learning_rate 2e-5 \
	--max_length ${maxlen} \
	--checkpoint_step 50 \
	--parallel \
	--amp \
	--random_seed ${rs} \
	--loss "infoNCE" \
	--infoNCE_tau ${infoce} \
	--pairwise \
	--dropout_rate ${dropout} \
	--agg_mode ${agg} \
 	# --save_checkpoint_all

