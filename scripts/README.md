BERT fine-tuning example:

`CUDA_VISIBLE_DEVICES=5 python early_stop_fine_tuning_bert.py --period 212 --save-stats 1 --bert-pooling 1 --save-model 1 --reset-layers 3 --layerwise-lr 0.9 --wandb 0 --epochs 20 --bert-size base --save-name llrd-20epochs-early-stop-reinit-3 --dataset 9`

Pegenerate data for BERT adaptive pretraining:

`python pregenerate_training_data.py --train_corpus ../data/processed/bert/bert_all.txt --output_dir ../data/processed/bert/pregenerate_data.txt --bert_model ../models/bert/base --max_seq_len 150 --epochs_to_generate 100`

Pretrain BERT with MLM and NSP:

`CUDA_VISIBLE_DEVICES=5 python tapt_bert_mlm_nsp.py --pregenerated_data ../data/processed/bert/pregenerate_data --output_dir ../models/bert/base-TAPT-MLM-NSP --bert_model ../models/bert/base --epochs 100 --adam_epsilon 1e-6 --learning_rate 1e-5`
