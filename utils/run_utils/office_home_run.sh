#!/bin/bash
domains=("p" "a" "c" "r")

for domain in "${domains[@]}"; do
    echo "Testing domain: $domain"
    python algorithms/feddg_moe/train_officehome.py --test_domain "$domain" --model clip_moe --batch_size 64 --local_epochs 5 --comm 40 --lr 0.001
done