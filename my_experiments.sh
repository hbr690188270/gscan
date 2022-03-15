# for test only
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_cpg_run1 --training_batch_size=200 --max_training_iterations=200000 --model_type cpg --seed=106 --cnn_kernel_size=13 --max_training_examples 100 &> target_lengths_cpg_run1/target_lengths_run.txt

# run1
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_cpg_run1 --training_batch_size=200 --max_training_iterations=200000 --model_type cpg --seed=106 --cnn_kernel_size=13 &> target_lengths_cpg_run1/target_lengths_run.txt

## baseline_no eos
python -m seq2seq --mode=train --max_decoding_steps=20 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/noeos_train --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=106 --cnn_kernel_size=13 &> checkpoints/noeos_train/train_log.txt

# run2
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_cpg_run2 --training_batch_size=200 --max_training_iterations=200000 --model_type cpg --seed=116 --cnn_kernel_size=13 &> target_lengths_cpg_run2/target_lengths_run.txt

# run2_test
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=100 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=test --training_batch_size=200 --max_training_iterations=200000 --max_training_examples 100 --model_type cpg --seed=116 --cnn_kernel_size=13 


# run3
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_cpg_run3 --training_batch_size=200 --max_training_iterations=200000 --model_type cpg --seed=126 --cnn_kernel_size=13 &> target_lengths_cpg_run3/target_lengths_run.txt


# baseline run2
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_2 --training_batch_size=200 --max_training_iterations=200000 --seed=116 --cnn_kernel_size=13 &> target_lengths_run_2/target_lengths_run.txt


# baseline inference
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --output_file_name=target_lengths_predict_run_3.json --max_decoding_steps=120 --cnn_kernel_size=13

# baseline inference 20000
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=baseline_inf_3 --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --output_file_name=target_lengths_predict_run_3.json --max_decoding_steps=120 --cnn_kernel_size=13 --max_testing_examples 10000

# baseline beam search 
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=baseline_inf_3 --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --output_file_name=target_lengths_predict_run_3.json --max_decoding_steps=120 --cnn_kernel_size=13 --inf_type beam_search --max_testing_examples 100


# cpg inference
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_cpg_run2 --resume_from_file=target_lengths_cpg_run2/model_best.pth.tar --splits=target_lengths --output_file_name=target_lengths_cpg_run_2.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type cpg

# eos distribution inference
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --output_file_name=eos_distribution.json --max_decoding_steps=120 --cnn_kernel_size=13 --inf_type eos_distribution





python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_cpg_run2 --resume_from_file=target_lengths_cpg_run2/model_best.pth.tar --splits=dev,test --output_file_name=target_lengths_cpg_run_2.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type cpg --max_testing_example 100


python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=test --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=dev,test --output_file_name=test.json --max_decoding_steps=120 --cnn_kernel_size=13 --max_testing_example 100

python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=test --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=dev --output_file_name=test.json --max_decoding_steps=120 --cnn_kernel_size=13 --max_testing_example 100



## transformer decoder1
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --no_auxiliary_task --conditional_attention --output_directory=transformer_length_splits --training_batch_size=200 --max_training_iterations=200000  --model_type transformer --seed=116 --cnn_kernel_size=13 --max_training_examples 100 --max_testing_example 100

## transformer decoder2
python -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --no_auxiliary_task --conditional_attention --output_directory=transformer_attentionv2 --training_batch_size=200 --max_training_iterations=200000  --model_type transformer --seed=116 --cnn_kernel_size=13 --learning_rate 0.001 &> transformer_attentionv2/train_log.txt

# transformer inf no eos
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=transformer_length_splits --resume_from_file=transformer_length_splits/model_best.pth.tar --splits=target_lengths --output_file_name=inf_noeos_transformer.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type transformer --inf_type no_eos

# transformer inf eos
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=transformer_length_splits --resume_from_file=transformer_length_splits/model_best.pth.tar --splits=target_lengths --output_file_name=inf_eos_transformer.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type transformer

## length predict
python lenpred_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --no_auxiliary_task --conditional_attention --output_directory=length_pred_transformer --training_batch_size=200 --max_training_iterations=200000  --model_type transformer --seed=116 --cnn_kernel_size=13 --max_training_examples 100 --max_testing_example 100

## length predict  transformer 6 layer
python lenpred_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --no_auxiliary_task --conditional_attention --output_directory=length_pred_transformer_layer6 --training_batch_size=200 --max_training_iterations=200000  --model_type transformer --seed=116 --cnn_kernel_size=13 --num_transformer_layers 6 &>length_pred_transformer_layer6/train_log.txt

## length predict  transformerv2 6 layer   note: still transformer   a typo
python lenpred_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --no_auxiliary_task --conditional_attention --encoder_hidden_size=128 --output_directory=length_pred_transformerv2_layer6 --training_batch_size=200 --max_training_iterations=200000  --model_type transformer --seed=116 --cnn_kernel_size=13 --num_transformer_layers 6  &>length_pred_transformerv2_layer6/train_log.txt

## length predict  transformerv2 1 layer
python lenpred_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --no_auxiliary_task --conditional_attention --encoder_hidden_size=128 --output_directory=length_pred_transformerv2_layer1 --training_batch_size=200 --max_training_iterations=200000  --model_type transformer --seed=116 --cnn_kernel_size=13 --num_transformer_layers 1  &>length_pred_transformerv2_layer1/train_log.txt


## length predict inference
python lenpred_inf.py --max_decoding_steps=120  --data_directory=data/target_length_split --no_auxiliary_task --conditional_attention --output_directory=length_pred_transformerv2_layer6 --resume_from_file=length_pred_transformerv2_layer6/model_best.pth.tar --model_type transformer --seed=116 --cnn_kernel_size=13 --splits=test,target_lengths --output_file_name=test.json --encoder_hidden_size=128 --num_transformer_layers 6 --max_testing_examples 100



## length bias

python lenbias_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=lenbias --training_batch_size=200 --max_training_iterations=200000 --seed=106 --cnn_kernel_size=13 --max_training_examples 100 --max_testing_example 100

python lenbias_train.py --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=lenbias --resume_from_file=lenbias/model_best.pth.tar --splits=target_lengths --output_file_name=lenbias_pred.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type cpg --max_testing_example 100

python lenbias_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=lenbias --training_batch_size=200 --max_training_iterations=200000 --seed=106 --cnn_kernel_size=13  --alpha=0.01 --max_training_examples 100 --max_testing_example 100

## normal  inf  with length reg
python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --max_decoding_steps=120 --cnn_kernel_size=13 --inf_type len_reg  --output_directory=baseline_lenreg_inf --alpha 0.1 --output_file_name=alpha0.1.json

python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --max_decoding_steps=120 --cnn_kernel_size=13 --inf_type len_reg --output_directory=baseline_lenreg_inf --alpha 1.0 --output_file_name=alpha1.0.json

python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --max_decoding_steps=120 --cnn_kernel_size=13 --inf_type len_reg  --output_directory=baseline_lenreg_inf --alpha 0.5 --output_file_name=alpha0.5.json

python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --max_decoding_steps=120 --cnn_kernel_size=13 --inf_type len_reg  --output_directory=baseline_lenreg_inf --alpha 0.05 --output_file_name=alpha0.05.json

python -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=target_lengths --max_decoding_steps=120 --cnn_kernel_size=13 --inf_type len_reg  --output_directory=baseline_lenreg_inf --alpha 0.01 --output_file_name=alpha0.01.json


# +PAD
python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.7 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.7 --white_portion=0.5 &>white_baseline_aug0.7/training_log.txt 

python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.5 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.5 --white_portion=0.5  &>white_baseline_aug0.5/training_log.txt 

python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.3 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.3 --white_portion=0.5 &>white_baseline_aug0.3/training_log.txt  

python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.1 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5  &>white_baseline_aug0.1/training_log.txt 


# PAD inference
python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.7  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=white_baseline_aug0.7/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 

python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.5  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=white_baseline_aug0.5/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 

python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.3  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=white_baseline_aug0.3/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 

python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=white_baseline_aug0.1  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=white_baseline_aug0.1/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 

## pad inference 2
python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_start_rand5_aug0.7  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=checkpoints/white_baseline_start_rand5_aug0.7/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 

python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_start_rand5_aug0.1  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=checkpoints/white_baseline_start_rand5_aug0.1/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 

python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_mid_rand5_aug0.7  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=checkpoints/white_baseline_mid_rand5_aug0.7/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 

python pad_train.py --mode=test --max_decoding_steps=120 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_mid_rand5_aug0.1  --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --resume_from_file=checkpoints/white_baseline_mid_rand5_aug0.1/model_best.pth.tar --splits=target_lengths --output_file_name=result.json 


## pad train 2

python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_start_rand5_aug0.7 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.7 --white_portion=0.5 --insertion start --aug_strategy rand --max_white_num 5 &>checkpoints/white_baseline_start_rand5_aug0.7/training_log.txt 

python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_start_rand5_aug0.1 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion start --aug_strategy rand --max_white_num 5 &>checkpoints/white_baseline_start_rand5_aug0.1/training_log.txt 


python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_mid_rand5_aug0.7 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.7 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 &>checkpoints/white_baseline_mid_rand5_aug0.7/training_log.txt 

python pad_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/white_baseline_mid_rand5_aug0.1 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 &>checkpoints/white_baseline_mid_rand5_aug0.1/training_log.txt 



## discriminator
python disc_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --max_training_examples 100 --max_testing_example 100


python disc_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc_layer6 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 &> checkpoints/disc_layer6/training_log.txt

python disc_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc_layer6_v2 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --data_type v2  &> checkpoints/disc_layer6_v2/training_log.txt

python disc_train.py --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc_layer6_v4 --training_batch_size=200 --max_training_iterations=200000 --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --data_type v4  &> checkpoints/disc_layer6_v4/training_log.txt

## discriminator inf
python disc_train.py --mode=test  --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc_layer6 --resume_from_file checkpoints/disc_layer6/model_best.pth.tar --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,test,target_lengths --output_file_name=result.json

python disc_train.py --mode=test  --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc_layer6_v2 --resume_from_file checkpoints/disc_layer6_v2/model_best.pth.tar --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,test,target_lengths --output_file_name=result.json  --data_type v2 --max_testing_example 10000

python disc_train.py --mode=test  --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc_layer6_v2 --resume_from_file checkpoints/disc_layer6_v2/model_best.pth.tar --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=target_lengths --output_file_name=result.json  --data_type v3 --max_testing_example 10000


## v4 inf

python disc_train.py --mode=test  --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=checkpoints/disc_layer6_v4 --resume_from_file checkpoints/disc_layer6_v4/model_best.pth.tar --model_type orig --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,test,target_lengths --output_file_name=result.json  --data_type v2 --max_testing_example 10000

## baseline inf + discriminator
python inf_with_disc.py --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=checkpoints/target_lengths_run_3/model_best.pth.tar --splits=target_lengths --output_file_name=inf_with_disc.json --max_decoding_steps=120 --cnn_kernel_size=13 --max_testing_examples 100

python inf_with_disc.py --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=checkpoints/target_lengths_run_3/model_best.pth.tar --splits=target_lengths --output_file_name=inf_with_disc.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type mc    --max_testing_examples 100

python inf_with_saved.py --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=checkpoints/target_lengths_run_3/model_best.pth.tar --splits=target_lengths --output_file_name=inf_with_disc.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type mc  --num_transformer_layers 6 --max_testing_examples 100

## baseline  noeos train inf
python inf_with_disc.py --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=checkpoints/noeos_train/model_best.pth.tar --splits=target_lengths --output_file_name=inf_with_disc.json --max_decoding_steps=120 --cnn_kernel_size=13 --model_type mc  --max_testing_examples 100


## multiple choice train  layer1
python disc_train.py --mode=train --max_decoding_steps=120 --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_2_2 --training_batch_size=80 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 2 --contrast_from_batch_size 2 &>checkpoints/disc_mc_2_2/train_log.txt

python disc_train.py --mode=train --max_decoding_steps=120 --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_20_20 --training_batch_size=10 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 &>checkpoints/disc_mc_20_20/train_log.txt


python disc_train.py --mode=train --max_decoding_steps=120 --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_20_20_continue --resume_from_file checkpoints/disc_mc_20_20/checkpoint.pth.tar --training_batch_size=60 --max_training_iterations=400000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 &>checkpoints/disc_mc_20_20_continue/train_log.txt

## multiple choice train  layer6  旧版本，默认control length
python disc_train.py --mode=train --max_decoding_steps=120 --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_20_20_layer6 --training_batch_size=30 --test_batch_size 30 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 &>checkpoints/disc_mc_20_20_layer6/train_log.txt

## multiple choice train layer 6  do not control length  新版本，默认不control length
python disc_train.py --mode=train --max_decoding_steps=120 --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_20_20_layer6_noLenControl --training_batch_size=30 --test_batch_size 30 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 &>checkpoints/disc_mc_20_20_layer6_noLenControl/train_log.txt

python disc_train.py --mode=train --max_decoding_steps=120 --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_10_10_layer4_noLenControl --training_batch_size=64 --test_batch_size 30 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --data_type mc --contrast_size 10 --contrast_from_batch_size 10 --num_transformer_layers 4 &>checkpoints/disc_mc_10_10_layer4_noLenControl/train_log.txt

python disc_train.py --mode=train --max_decoding_steps=120 --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_20_20_layer4_noLenControl --training_batch_size=64 --test_batch_size 30 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 4 &>checkpoints/disc_mc_20_20_layer4_noLenControl/train_log.txt

## mc inference layer6
python disc_train.py --mode=test  --data_directory=data/target_length_split  --output_directory=checkpoints/disc_mc_20_20_layer6 --resume_from_file checkpoints/disc_mc_20_20_layer6/model_best.pth.tar --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,target_lengths --output_file_name=result.json  --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --max_testing_example 10000

python disc_train.py --mode=test  --data_directory=data/target_length_split  --output_directory=checkpoints/disc_mc_20_20_layer6 --resume_from_file checkpoints/gan_0.01/model_best.pth.tar --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,target_lengths --output_file_name=result.json  --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --max_testing_example 10000

python disc_train.py --mode=test  --data_directory=data/target_length_split  --output_directory=checkpoints/tri_stage_0.01 --resume_from_file checkpoints/tri_stage_0.01/acc0.689_l152.011 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,target_lengths --output_file_name=result.json  --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --max_testing_example 10000

python disc_train.py --mode=test  --data_directory=data/target_length_split  --output_directory=checkpoints/tri_stage_0.01 --resume_from_file checkpoints/tri_stage_1.0_cotrain/acc0.614_l16.574 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,target_lengths --output_file_name=result.json  --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --max_testing_example 10000

python disc_train.py --mode=test  --data_directory=data/target_length_split  --output_directory=checkpoints/tri_stage_0.01 --resume_from_file checkpoints/tri_stage_0.5_cotrain/acc0.694_l112.241 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,target_lengths --output_file_name=result.json  --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --max_testing_example 10000


python disc_train.py --mode=test  --data_directory=data/target_length_split  --output_directory=checkpoints/disc_mc_20_20_layer6_noLenControl --resume_from_file checkpoints/disc_mc_20_20_layer6_noLenControl/model_best.pth.tar --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,test,target_lengths --output_file_name=result.json  --data_type mc --max_testing_example 10000

python disc_train.py --mode=test  --data_directory=data/target_length_split  --output_directory=checkpoints/disc_mc_20_20_layer6_noLenControl --resume_from_file checkpoints/gan_cls_0.1/model_best.pth.tar --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --num_transformer_layers 6 --splits=dev,target_lengths --output_file_name=result.json  --data_type mc --max_testing_example 10000


## sanity check
python sanity_check.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/sanity_check --training_batch_size=64 --test_batch_size 30 --max_training_iterations=200000 --seed=126 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 4 &>checkpoints/sanity_check/train_log.txt

python sanity_check.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/sanity_check --training_batch_size=10 --test_batch_size 10 --max_training_iterations=200000 --seed=126 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --contrast_size 2 --contrast_from_batch_size 2 --num_transformer_layers 4 --max_training_examples 10 --max_testing_example 10 --print_every=5

## disc+gan
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_test --training_batch_size=30 --test_batch_size 30 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --max_training_examples 100 --max_testing_example 100

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_0.01 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 0.01 &> checkpoints/gan_0.01/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_0.1_smoothl1 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 0.1 &> checkpoints/gan_0.1_smoothl1/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_0.05 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 0.05 &> checkpoints/gan_0.05/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_0.5 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 0.5 &> checkpoints/gan_0.5/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_1.0_smoothl1 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 1.0 &> checkpoints/gan_1.0_smoothl1/train_log.txt

## disc + gan + classification
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_cls_1.0 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 1.0 &> checkpoints/gan_cls_1.0/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_cls_0.5 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 0.5 &> checkpoints/gan_cls_0.5/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_cls_0.1 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 0.1 &> checkpoints/gan_cls_0.1/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_cls_0.01 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --training_type gan --adv_beta 0.01 &> checkpoints/gan_cls_0.01/train_log.txt


## tri_stage gan reg
python tri_stage_gan_main.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/tri_stage_0.01_cotrain --training_batch_size=12 --test_batch_size=12 --max_training_iterations=300000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --warmup 1000 --adv_beta 0.01 --resume_from_file checkpoints/disc_mc_20_20_layer6/model_best.pth.tar --start_stage 2 &> checkpoints/tri_stage_0.01_cotrain/train_log.txt

python tri_stage_gan_main.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/tri_stage_1.0_cotrain --training_batch_size=12 --test_batch_size=12 --max_training_iterations=300000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --warmup 1000 --adv_beta 1.0 --resume_from_file checkpoints/disc_mc_20_20_layer6/model_best.pth.tar --start_stage 2 &> checkpoints/tri_stage_1.0_cotrain/train_log.txt
(screen 0.01)

python tri_stage_gan_main.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/tri_stage_0.5_cotrain --training_batch_size=12 --test_batch_size=12 --max_training_iterations=300000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --warmup 1000 --adv_beta 0.5 --resume_from_file checkpoints/disc_mc_20_20_layer6/model_best.pth.tar --start_stage 2 &> checkpoints/tri_stage_0.5_cotrain/train_log.txt


python tri_stage_gan_main.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/tri_stage_0.1 --training_batch_size=16 --test_batch_size=16 --max_training_iterations=300000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --warmup 1000 --adv_beta 0.1 --resume_from_file checkpoints/disc_mc_20_20_layer6/model_best.pth.tar --start_stage 2 &> checkpoints/tri_stage_0.1/train_log.txt

python tri_stage_gan_main.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/tri_stage_0.5 --training_batch_size=12 --test_batch_size=12 --max_training_iterations=300000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 6 --length_control --warmup 1000 --adv_beta 0.5 --resume_from_file checkpoints/disc_mc_20_20_layer6/model_best.pth.tar --start_stage 2 &> checkpoints/tri_stage_0.5/train_log.txt


python main_simple.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/simplev1 --training_batch_size=60 --test_batch_size 60 --max_training_iterations=200000  --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --model_type v1 --max_training_examples 100 --max_testing_examples 100

python main_simple.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/simplev2 --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000  --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --model_type v2

python main_simple.py --mode=test --data_directory=data/target_length_split --resume_from_file=checkpoints/simplev2/model_best.pth.tar --training_batch_size=24 --test_batch_size 12 --max_training_iterations=200000  --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --splits=dev,target_lengths --model_type v2 --max_testing_example 10000

## disc mc v2
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_v2 --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control &> checkpoints/disc_mc_v2/train_log.txt

## disc mc v3
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_v3 --training_batch_size=40 --test_batch_size 40 --max_training_iterations=200000 --model_type mc_v3 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control &> checkpoints/disc_mc_v3/train_log.txt

## disc mc v2  use_gan   adv_beta 0
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_test --training_batch_size=64 --test_batch_size 64 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --training_type gan --adv_beta 0.0  &>checkpoints/gan_test/train_log.txt

## disc mc 原始版本  use_gan   adv_beta 0  classification
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_model_v1_cls --training_batch_size=24 --test_batch_size 24 --max_training_iterations=200000 --model_type mc --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 4 --length_control --training_type gan --adv_beta 0.0  &>checkpoints/gan_model_v1_cls/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/gan_model_v2_reg --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --training_type gan --adv_beta 0.0  --loss_type reg &>checkpoints/gan_model_v2_reg/train_log.txt


### mc sanity check用
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/test --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --max_training_examples 1 --warmup 1

### gan+cls  用一个训练好的mc模型，只训练cls
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/mc_adv_only --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --adv_beta 0.0 --training_type adv_only --resume_from_file checkpoints/prev/disc_mc_v2_dropout0/model_best.pth.tar &>checkpoints/mc_adv_only/train_log.txt

### gan+cls  简易分类，5类
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/mc_easycls_gan --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --adv_beta 0.0 --training_type gan --less_length_label  &>checkpoints/mc_easycls_gan/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/mc_easycls_adv_only --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --adv_beta 0.0 --training_type adv_only --less_length_label --resume_from_file checkpoints/disc_mc_v2/model_best.pth.tar &>checkpoints/mc_easycls_adv_only/train_log.txt

## dropout 0
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_v2_dropout0 --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --dropout 0.0 &> checkpoints/disc_mc_v2_dropout0/train_log.txt

## lr /=5
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_v2_lr0.0002 --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --learning_rate 0.0002 &> checkpoints/disc_mc_v2_lr0.0002/train_log.txt

## 只使用batch negative
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_v2_no_mannual_contrast --training_batch_size=48 --test_batch_size 48 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 0 --contrast_from_batch_size 40 --num_transformer_layers 2 --length_control &> checkpoints/disc_mc_v2_no_mannual_contrast/train_log.txt

python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_v2_bsz60 --training_batch_size=60 --test_batch_size 60 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control  --learning_rate 0.0005 &> checkpoints/disc_mc_v2_bsz60/train_log.txt

## 问题已经修复，沿用之前的setting
python main_disc.py --mode=train --data_directory=data/target_length_split --output_directory=checkpoints/disc_mc_v2 --training_batch_size=32 --test_batch_size 32 --max_training_iterations=200000 --model_type mc_v2 --seed=126 --cnn_kernel_size=13 --target_vocab_path=training_target_vocab_white.txt --input_vocab_path=training_input_vocab_white.txt --aug_prob=0.1 --white_portion=0.5 --insertion mid --aug_strategy rand --max_white_num 5 --data_type mc --contrast_size 20 --contrast_from_batch_size 20 --num_transformer_layers 2 --length_control --dropout 0.1 --evaluate_every 2000 &> checkpoints/disc_mc_v2/train_log.txt



