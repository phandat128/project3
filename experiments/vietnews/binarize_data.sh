src=article
tgt=abstract
PROJ=.
INPUT=$PROJ/data/vietnews
OUTPUT=$INPUT/bin

fairseq-preprocess \
	--source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/train.short.tok \
	--validpref $INPUT/validation.short.tok,$INPUT/validation.long.tok \
	--testpref $INPUT/test.short.tok,$INPUT/test.long.tok \
	--destdir $OUTPUT \
	--workers 2 \
	--joined-dictionary