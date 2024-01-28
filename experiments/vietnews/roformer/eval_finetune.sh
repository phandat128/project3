export CUDA_VISIBLE_DEVICES=0
src=article
tgt=abstract
PROJ_PATH=./experiments/vietnews
DATA_PATH=./data/vietnews/finetune-bin
CKPT_PATH=$PROJ_PATH/roformer/finetune_train_log
MODEL_DIR=$PROJ_PATH/roformer
OUTPUT_FN=$MODEL_DIR/res.txt

mkdir -p $MODEL_DIR/finetuned_outputs

for split in test test1; do
  fairseq-generate $DATA_PATH \
          --gen-subset $split \
          --path $CKPT_PATH/checkpoint_last.pt \
          --max-tokens 4096 \
  	      --remove-bpe \
          --beam 4 \
          --lenpen 0.6 \
          --skip-invalid-size-inputs-valid-test \
          > $OUTPUT_FN

  # Extract source, predictions and ground truth
  grep '^S-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f2 > $MODEL_DIR/finetuned_outputs/src.tok.$split
  grep '^H-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f3 > $MODEL_DIR/finetuned_outputs/preds.tok.$split
  grep '^T-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f2 > $MODEL_DIR/finetuned_outputs/truth.tok.$split

  # Fix some moses detokenization
  sed "s| '|'|g" $MODEL_DIR/finetuned_outputs/preds.tok.$split | sed "s| /|/|g" | sed "s|/ |/|g" | sed "s| @ - @ |-|g" \
  	  > $MODEL_DIR/finetuned_outputs/preds.$split
  sed "s| '|'|g" $MODEL_DIR/finetuned_outputs/truth.tok.$split | sed "s| /|/|g" | sed "s|/ |/|g" | sed "s| @ - @ |-|g" \
          > $MODEL_DIR/finetuned_outputs/truth.$split
  rm $MODEL_DIR/finetuned_outputs/preds.tok.$split $MODEL_DIR/finetuned_outputs/truth.tok.$split

done

rm $OUTPUT_FN