mkdir -p ./saved_models/cache_data
CUDA_VISIBLE_DEVICES=0 python run_gen.py    \
    --do_test \
    --save_last_checkpoints \
    --always_save_model   \
    --task summarize \
    --sub_task java \
    --model_type codet5 \
    --data_num -1   \
    --num_train_epochs 10 \
    --warmup_steps 1000 \
    --learning_rate 5e-5 \
    --patience 3   \
    --tokenizer_name=Salesforce/codet5-base \
    --load_model_path=codet5/saved_models/checkpoint-best-bleu/pytorch_model.bin \
    --tokenizer_path=../../../CodeT5/tokenizer/salesforce   \
    --model_name_or_path=Salesforce/codet5-base \
    --train_filename=data/java/train.jsonl \
    --dev_filename=data/java/valid.jsonl \
    --test_filename=data/java/test.jsonl \
    --output_dir saved_models/  \
    --summary_dir tensorboard   \
    --data_dir ../data/  \
    --cache_path saved_models/cache_data \
    --res_dir saved_models/prediction \
    --res_fn saved_models/summarize_codet5_base.txt   \
    --train_batch_size 28 \
    --eval_batch_size 12 \
    --max_source_length 256 \
    --max_target_length 128  \
    2>&1 | tee saved_models/train.log
