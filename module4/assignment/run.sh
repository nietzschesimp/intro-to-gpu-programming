#!/bin/bash

# This is the code that runs the testbed comparing memory execution


NORUNS=5
KEYWORD_TIME_W_TRANSFER="\{.*\}"
KEYWORD_TIME_NO_TRANSFER="\(.*\)"

OPS="add sub mul mod"

NOTHREADS="32 64 128 256 512 1024"
INSIZE=131072

echo "----------------------------------------------------------"
echo "------------------------- PINNED -------------------------"
echo "----------------------------------------------------------"

for OP in $OPS
do
	echo "------------------------- $OP -------------------------" 

	for THREAD in $NOTHREADS
	do
		echo "Number of inputs: $INSIZE"
		echo "Number of threads: $THREAD"
		echo "Group size: $(expr $INSIZE / $THREAD)"
		
		rm /tmp/benchmark

		for i in $(seq 1 $NORUNS)
		do
			./src/pinned_memory $(expr $INSIZE / $THREAD) $INSIZE $OP >> /tmp/benchmark
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
	done
done

echo "---------------------------------------------------------"
echo "------------------------- PAGED -------------------------"
echo "---------------------------------------------------------"

for OP in $OPS
do
	echo "------------------------- $OP -------------------------" 

	for THREAD in $NOTHREADS
	do
		echo "Number of inputs: $INSIZE"
		echo "Number of threads: $THREAD"
		echo "Group size: $(expr $INSIZE / $THREAD)"
		
		rm /tmp/benchmark

		for i in $(seq 1 $NORUNS)
		do
			./src/host_memory $(expr $INSIZE / $THREAD) $INSIZE $OP >> /tmp/benchmark
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
	done
done

echo "---------------------------------------------------------"
echo "-------------------------- CPU --------------------------"
echo "---------------------------------------------------------"

for OP in $OPS
do
	echo "------------------------- $OP -------------------------" 

	echo "Number of inputs: $INSIZE"
	
	rm /tmp/benchmark

	for i in $(seq 1 $NORUNS)
	do
		./src/cpu_proc $INSIZE $OP >> /tmp/benchmark
	done

	AVG_TIME=0
	for TIME in $(cat /tmp/benchmark | grep -oE $KEYWORD_TIME_W_TRANSFER | grep -oE "[0-9]+.[0-9]+")
	do
		AVG_TIME=$(echo "$AVG_TIME + $TIME" | bc -l)
	done

	AVG_TIME=$(echo "$AVG_TIME / $NORUNS" | bc -l)

	echo "Average time to execute: $AVG_TIME ms"
done
