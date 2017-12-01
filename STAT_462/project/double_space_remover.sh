#!/bin/bash
rm clean.cu
perl -00 -pe 1 project_no_transform.R > clean.cu
