DATADIR=./data/vietnews/finetune-bin
CKPTS=./experiments/vietnews/roformer/train_log/checkpoint_last.pt
FT_CKPTS=./experiments/vietnews/roformer/finetune_train_log

params="$DATADIR \
--num-workers 2 \
--ignore-unused-valid-subsets \
--save-dir $FT_CKPTS \
--restore-file $CKPTS \
--reset-dataloader \
--reset-lr-scheduler \
--reset-meters \
--reset-optimizer \
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
--lr 1e-4 \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--max-update 1000 \
--no-progress-bar \
--log-format json \
--log-interval 100 \
--save-interval 500000 \
--save-interval-updates 500 \
--keep-interval-updates 1 \
--update-freq 4 \
--rotary-embedding \
--scaling-type PI \
--scaling-factor 2.0 \
--fp16
"

mkdir -p $CKPTS

fairseq-train $params
