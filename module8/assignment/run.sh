#!/bin/bash

# This is the code that runs the testbed comparing memory execution

if [[ ! -d "build" ]]
then
	echo "Project is not built, run build.sh"
	exit -1
fi

NORUNS=5
KEYWORD_TIME_W_TRANSFER="\{.*\}"
KEYWORD_TIME_NO_TRANSFER="\(.*\)"

NO_ROWS=1920
NO_COLS=1080
SCALAR=5.0

echo "----------------------------------------------------------------"
echo "| CONVOLUTION"
echo "----------------------------------------------------------------"
echo "rows: $NO_ROWS"
echo "cols: $NO_COLS"

rm /tmp/benchmark

for i in $(seq 1 $NORUNS)
do
	./build/src/convolution2d -r $NO_COLS -c $NO_ROWS >> /tmp/benchmark
done

AVG_TIME=0
for TIME in $(cat /tmp/benchmark | grep -oE $KEYWORD_TIME_NO_TRANSFER | grep -oE "[0-9]+.[0-9]+")
do
	AVG_TIME=$(echo "$AVG_TIME + $TIME" | bc -l)
done

AVG_TIME=$(echo "$AVG_TIME / $NORUNS" | bc -l)

echo "Average time to execute convolution: $AVG_TIME ms"

echo "----------------------------------------------------------------"
echo "| MATRIX SCALING"
echo "----------------------------------------------------------------"
echo "scalar: $SCALAR"
echo "rows: $NO_ROWS"
echo "cols: $NO_COLS"

rm /tmp/benchmark

for i in $(seq 1 $NORUNS)
do
	./build/src/matrix_scale -r $NO_COLS -c $NO_ROWS -s $SCALAR >> /tmp/benchmark
done

AVG_TIME=0
for TIME in $(cat /tmp/benchmark | grep -oE $KEYWORD_TIME_NO_TRANSFER | grep -oE "[0-9]+.[0-9]+")
do
	AVG_TIME=$(echo "$AVG_TIME + $TIME" | bc -l)
done

AVG_TIME=$(echo "$AVG_TIME / $NORUNS" | bc -l)

echo "Average time to execute scaling: $AVG_TIME ms"
