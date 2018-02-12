import numpy as np


def findLongestSequence(array_in):
    current_element = array_in[0]
    longest_sequence = -1
    current_sequence = 1

    for i in range(1, len(array_in)):
        if array_in[i] == current_element:
            current_sequence += 1
        else:
            if current_sequence > longest_sequence:
                longest_sequence = current_sequence
            current_sequence = 1
        current_element = array_in[i]

    if current_sequence > longest_sequence:
        longest_sequence = current_sequence

    return longest_sequence


# Question 1
def q1(history):
    iter = 1000000
    longest_in = findLongestSequence(history)
    print(longest_in)
    equal_length_successes = 0

    for x in range(0, iter):
        np.random.shuffle(history)
        longest_seq = findLongestSequence(history)
        if longest_seq >= longest_in:
            equal_length_successes += 1

    return (equal_length_successes / iter)


# Question 2
def q2(p):
    None


# Question 3
def q3(N):
    None