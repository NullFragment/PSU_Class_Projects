#!/bin/bash
./final < inputs/create_tables.txt > outputs/create_tables.out
./final < inputs/insert_most.txt > outputs/insert_most.out
./final < inputs/insert_shifts.txt > outputs/insert_shifts.out
./final < inputs/insert_needs.txt > outputs/insert_needs.out
