#!/bin/bash
clear
nvcc cuda_nn.cu -lcublas -o main
./main
