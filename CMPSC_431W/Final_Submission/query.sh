#!/bin/bash
rm -f outputs/query*.out
./final < inputs/query1.txt > outputs/query1.out
./final < inputs/query2.txt > outputs/query2.out
./final < inputs/query3.txt > outputs/query3.out
./final < inputs/query4.txt > outputs/query4.out

