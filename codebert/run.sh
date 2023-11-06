lang=java #programming language
mkdir -p ./saved_models/$lang/
CUDA_VISIBLE_DEVICES=3 python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --train_filename ../data/$lang/train.jsonl \
        --dev_filename ../data/$lang/valid.jsonl \
        --test_filename ../data/$lang/test.jsonl \
        --output_dir ./saved_models/$lang \
        --max_source_length 192 \
        --max_target_length 64 \
        --beam_size 10 \
        --train_batch_size 40 \
        --eval_batch_size 32 \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        2>&1 | tee ./saved_models/$lang/train.log