DATADIR=./data/nc15ende/en2de
CKPTS=./experiments/nc15en2de/transformer/train_log

params="$DATADIR \
--num-workers 2 \
--save-dir $CKPTS \
--arch transformer \
--dropout 0.3 \
--share-all-embeddings \
--optimizer adam \
--adam-betas (0.9,0.98) \
--adam-eps 1e-09 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-07 \
--warmup-updates 8000 \
--lr 5e-4 \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 4096 \
--max-update 20000 \
--no-progress-bar \
--log-format json \
--log-interval 100 \
--save-interval 500000 \
--save-interval-updates 500 \
--keep-interval-updates 1 \
--fp16 \
--update-freq 4 \
"

mkdir -p $CKPTS

fairseq-train $params

read -p "exit"
