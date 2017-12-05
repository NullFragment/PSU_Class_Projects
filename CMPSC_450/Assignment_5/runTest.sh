#!/bin/bash
echo -e "N\tCPU\tSCAN\tFULL"
for ((m=1000; m < 10000000; m = m + 1000))
do
    ./assign_5 $m
done
