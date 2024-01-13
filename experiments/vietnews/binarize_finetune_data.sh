src=article
tgt=abstract
PROJ=.
INPUT=$PROJ/data/vietnews
OUTPUT=$INPUT/finetune-bin

fairseq-preprocess \
	--source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/train.long.tok \
	--validpref $INPUT/validation.short.tok,$INPUT/validation.long.tok \
	--testpref $INPUT/test.short.tok,$INPUT/test.long.tok \
	--tgtdict $INPUT/bin/dict.$tgt.txt \
	--srcdict $INPUT/bin/dict.$src.txt \
	--destdir $OUTPUT \
	--workers 2