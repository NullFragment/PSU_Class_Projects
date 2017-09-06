#!/bin/bash

g++ main.cpp -o main_default
g++ main.cpp -o main_O1 -O1
g++ main.cpp -o main_O2 -O2
g++ main.cpp -o main_O3 -O3
g++ main.cpp -o main_O1_mavx -O1 -mavx
g++ main.cpp -o main_O2_mavx -O2 -mavx
g++ main.cpp -o main_O3_mavx -O3 -mavx

echo -n "main_default   "
./main_default
echo -n "main_O1        "
./main_O1 1000 100
echo -n "main_O2        "
./main_O2 1000 100
echo -n "main_O3        "
./main_O3 1000 100
echo -n "main_O1_mavx   "
./main_O1_mavx 1000 100
echo -n "main_O2_mavx   "
./main_O2_mavx 1000 100
echo -n "main_O3_mavx   "
./main_O3_mavx 1000 100
echo "Done"
