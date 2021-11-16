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
INSIZE=65536

function run_test() {
	echo "----------------------------------------------------------------"
	echo "| $1" 
	echo "----------------------------------------------------------------"

	echo "No items in vector: $INSIZE"
	echo "Number of runs: $NORUNS"

	rm /tmp/benchmark

	for i in $(seq 1 $NORUNS)
	do
		$2 $3 >> /tmp/benchmark
	done

	AVG_TIME_NO_TRANSFER=0
	for TIME in $(cat /tmp/benchmark | grep -oE $KEYWORD_TIME_NO_TRANSFER | grep -oE "[0-9]+.[0-9]+")
	do
		AVG_TIME_NO_TRANSFER=$(echo "$AVG_TIME_NO_TRANSFER + $TIME" | bc -l)
	done

	AVG_TIME_W_TRANSFER=0
	for TIME in $(cat /tmp/benchmark | grep -oE $KEYWORD_TIME_W_TRANSFER | grep -oE "[0-9]+.[0-9]+")
	do
		AVG_TIME_W_TRANSFER=$(echo "$AVG_TIME_W_TRANSFER + $TIME" | bc -l)
	done

	AVG_TIME_NO_TRANSFER=$(echo "$AVG_TIME_NO_TRANSFER / $NORUNS" | bc -l)
	AVG_TIME_W_TRANSFER=$(echo "$AVG_TIME_W_TRANSFER / $NORUNS" | bc -l)

	echo "Average time to execute without transfer: $AVG_TIME_NO_TRANSFER ms"
	echo "Average time to execute with transfer: $AVG_TIME_W_TRANSFER ms"
}

# Test thrift operations
run_test "Convolution" "./build/src/convolution" "-c src/Convolution.cl -s src/Scaling.cl"
