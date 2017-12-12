#!/bin/bash
rm -f outputs/query*.output
./final < inputs/query1.txt > outputs/query1.output
./final < inputs/query2.txt > outputs/query2.output
./final < inputs/query3.txt > outputs/query3.output
./final < inputs/query4.txt > outputs/query4.output
