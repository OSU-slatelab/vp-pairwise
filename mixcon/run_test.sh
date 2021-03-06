python -u main.py \
    --word-vector "glove/vp18.glove.42B.300d.txt.pt" \
    --label-data "data/vp_18/vp.full.labels.json" \
    --traindev-data "data/vp_18/vp.full.train.json" \
    --dictionary "data/vp_18/vp.full.dict.json" \
    --test-data "data/vp_18/vp.full.test.json" \
    --attention-hops 2 \
    --nhid 300 \
    --epochs 4 \
    --stage2 2 \
    --batch-size 64 \
    --test-bsize 32 \
    --nclasses 374 \
    --optimizer "Adam" \
    --lr 0.00004 \
    --penalization-coeff 0.00 \
    --margin-pos 0.8 \
    --margin-neg 1.2 \
    --beta-max 10.0 \
    --beta-min 0.5 \
    --num-pos 50000 \
    --num-keys 3000 \
    --lamb 0.15 \
    --ploss_wt 0.15 \
    --prebert-path "/homes/3/sunder.9/vp-vishal/bert/pretrained_models/" \
    --bert-pooling "mean" \
    --seed 1111 \
    --cuda \
    --rsamp \
    --encoder-type "bert"
