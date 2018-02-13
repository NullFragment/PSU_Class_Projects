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

    iterations = 10000000
    longest_in = findLongestSequence(history)
    equal_length_successes = 0

    for x in range(0, iterations):
        np.random.shuffle(history)
        longest_seq = findLongestSequence(history)
        if longest_seq >= longest_in:
            equal_length_successes += 1

    return (equal_length_successes / iterations)


# Question 2
def q2(p):
    return(0)


# Question 3
def q3(N):
    assert (N % 2 == 0)
    seasons = 1000000
    pairs = N // 2
    teams = list(range(0, N))
    game_probabilities = [0.5] * pairs

    season_winners = [0] * seasons

    for season in range(0, seasons):
        wins = [0] * N

        for game in range(0, 16):
            np.random.shuffle(teams)
            winners = np.random.binomial(1, game_probabilities)

            for pair in range(0, pairs):
                if winners[pair] > 0.5:
                    wins[teams[pair * 2]] += 1
                else:
                    wins[teams[pair * 2 + 1]] += 1

        season_winners[season] = max(wins)

    return np.mean(season_winners)


print(q1([1, 1, 1, 0, 0, 0, 1, 0, 1, 0]))
print(q1([1, 1, 0, 0, 1, 1, 0, 0, 1, 1]))