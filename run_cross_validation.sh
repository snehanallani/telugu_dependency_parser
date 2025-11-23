#!/bin/bash
END=10
data_path=/home/sneha.nallani/old_tb/cross_validation_data/cv_data
vectors_path=/home/sneha.nallani/old_tb/cross_validation_data/cv_bert_vectors
input_data_dir=/scratch/cv_input_data
output_data_dir=./cv_output_inter_s0s1b0
train_suffix="_train.conll"
dev_suffix="_dev.conll"
test_suffix="_test.conll"
for i in $(seq 1 $END); 
do
    echo "Running set$i";
    cp "${data_path}/set$i$train_suffix" $input_data_dir/train.conll
    cp "${data_path}/set$i$dev_suffix" $input_data_dir/dev.conll
    cp "${data_path}/set$i$test_suffix" $input_data_dir/test.conll

    dev_set=$(($i + 8))
    test_set=$(($i + 9))
    if [ $dev_set -gt 10 ]; then
        dev_set=$(($dev_set - 10))
    fi
    if [ $test_set -gt 10 ]; then
        test_set=$(($test_set - 10))
    fi
    #echo "Dev set $dev_set"
    #echo "Test set $test_set"

    cp "${vectors_path}/set${i}_train_sentences_bert_vectors.jsonl" $input_data_dir/train_bert_vectors.jsonl
    cp "${vectors_path}/${dev_set}_sentences_bert_vectors.jsonl" $input_data_dir/dev_bert_vectors.jsonl
    cp "${vectors_path}/${test_set}_sentences_bert_vectors.jsonl" $input_data_dir/test_bert_vectors.jsonl

    #python3 run_parser.py
    python3 run_parser.py > "${output_data_dir}/set${i}_out"

    rm $input_data_dir/*

done
