#!/bin/bash
rm clean.cu
perl -00 -pe 1 cuda_nn.cu > clean.cu
