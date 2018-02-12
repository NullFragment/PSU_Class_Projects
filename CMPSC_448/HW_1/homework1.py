import numpy as np


def findLongestInitialSequence(array_in):
    current_element = array_in[0]
    to_match = current_element
    longest_sequence = -1
    current_sequence = 1

    for i in range(1, len(array_in)):
        if array_in[i] == current_element:
            current_sequence += 1
        else:
            if current_sequence > longest_sequence:
                longest_sequence = current_sequence
                to_match = current_element
            current_sequence = 1
        current_element = array_in[i]

    if current_sequence > longest_sequence:
        longest_sequence = current_sequence

    return longest_sequence, to_match


def findLongestSequence(array_in, to_match):
    longest_sequence = -1
    current_sequence = 0

    for i in range(0, len(array_in)):
        if array_in[i] == to_match:
            current_sequence += 1
        else:
            if current_sequence > longest_sequence:
                longest_sequence = current_sequence
            current_sequence = 0

    if current_sequence > longest_sequence:
        longest_sequence = current_sequence

    return longest_sequence


# Question 1
def q1(history):
    iterations = 10000
    longest_in, char_to_match = findLongestInitialSequence(history)
    print(longest_in)
    equal_length_successes = 0

    for x in range(0, iterations):
        np.random.shuffle(history)
        longest_seq = findLongestSequence(history, char_to_match)
        if longest_seq >= longest_in:
            equal_length_successes += 1

    return (equal_length_successes / iterations)
    

# Question 2
def q2(p):
    None


# Question 3
def q3(N):
    None

