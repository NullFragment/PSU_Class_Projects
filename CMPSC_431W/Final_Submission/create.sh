#!/bin/bash
rm -f outputs/create_tables.output
rm -f outputs/insert*
./final < inputs/create_tables.txt > outputs/create_tables.output
./final < inputs/insert_most.txt > outputs/insert_most.output
./final < inputs/insert_shifts.txt > outputs/insert_shifts.output
./final < inputs/insert_needs.txt > outputs/insert_needs.output
