for i in 0.00 0.15 0.25 0.5 0.75 1.00
do
    for j in 0.00 0.15 0.25 0.5 0.75 1.00
    do
        python -u main.py \
            --word-vector "glove/vp18.glove.42B.300d.txt.pt" \
            --label-data "data/vp_18/vp.full.labels.json" \
            --traindev-data "data/vp_18/vp.full.train.json" \
            --dictionary "data/vp_18/vp.full.dict.json" \
            --test-data "" \
            --valid-data "data/vp_18/vp.full.valid.json" \
            --attention-hops 2 \
            --epochs 12 \
            --stage2 3 \
            --nclasses 374 \
            --batch-size 512 \
            --test-bsize 32 \
            --penalization-coeff 0.0 \
            --margin-pos 0.8 \
            --margin-neg 1.2 \
            --beta-max 10.0 \
            --beta-min 0.5 \
            --lr 0.00004 \
            --num-pos 50000 \
            --validation-log "logs/vp_new_rnn.log" \
            --num-keys 3000 \
            --lamb $i \
            --ploss_wt $j \
            --encoder-type "rnn" \
            --prebert-path "bert/pretrained_models/" \
            --bert-pooling "mean" \
            --seed 1111 \
            --rsamp \
            --validation \
            --cuda 
    done
done
