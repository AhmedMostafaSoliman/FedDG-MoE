#!/bin/bash

# Define experiment parameters
domains=("p" "a" "c" "r")
trackers=(
    "online_gaussian"
    "online_cosine"
    "online_gmm"
    "offline_gmm"
    "offline_mahalanobis"
    "offline_cosine"
)
train_batch_sizes=(64 32 16)
test_batch_sizes=(64 32 16 8 4)

# Iterate over all combinations
for domain in "${domains[@]}"; do
    for tracker in "${trackers[@]}"; do
        for train_batch in "${train_batch_sizes[@]}"; do
            for test_batch in "${test_batch_sizes[@]}"; do
                echo "Running experiment:"
                echo "Domain: $domain"
                echo "Tracker: $tracker"
                echo "Train batch size: $train_batch"
                echo "Test batch size: $test_batch"
                
                python algorithms/feddg_moe/train_officehome.py \
                    --test_domain "$domain" \
                    --model clip_moe \
                    --batch_size "$train_batch" \
                    --test_batch_size "$test_batch" \
                    --domain_tracker "$tracker" \
                    --local_epochs 5 \
                    --comm 40 \
                    --lr 0.001 \
                    --note "${tracker}_train${train_batch}_test${test_batch}"
                
                echo "----------------------------------------"
            done
        done
    done
done

