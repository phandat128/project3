src=en
tgt=de
PROJ=.
INPUT=$PROJ/data/nc15ende
OUTPUT=$INPUT/${src}2${tgt}

fairseq-preprocess \
	--source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/train.tok \
	--validpref $INPUT/valid.tok \
	--testpref $INPUT/test.tok,$INPUT/long_test.tok \
	--destdir $OUTPUT \
	--workers 8 \
	--nwordssrc 16384 \
	--nwordstgt 16384 \
	--joined-dictionary