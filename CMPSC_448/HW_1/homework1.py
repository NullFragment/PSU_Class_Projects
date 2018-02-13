import numpy as np

# set number of iterations for all simulations:
iterations = 100000


def findPairedValues(array_in):
    pairs = 0

    # Simply iterate through history and find number of paired values
    for item in range(1, len(array_in)):
        if array_in[item] == array_in[item - 1]:
            pairs += 1

    return pairs


# Question 1
def q1(history):
    initial_pairs = findPairedValues(history)  # Store number of paired values to compare to for test statistic
    test_stat = 0

    for x in range(0, iterations):
        np.random.shuffle(history)  # Permute history
        pairs = findPairedValues(history)  # Find number of paired values in permutation
        if pairs >= initial_pairs:
            test_stat += 1  # Increment test statistic if success

    return test_stat / iterations  # Return p-value


# Question 2
def q2(p):
    # This is the result of solving for alpha (a) in the following costs:
    # Cost(treating, type A)*P(A) + Cost(treating, B)*P(B) = Cost(no treatment, A)*P(A) + Cost(no treatment, B)*P(B)
    # a*p + 1*(1-p) = 2*p + 5*(1-p)
    # a*p = 2*p + 4*(1-p)
    # a = [2*p + 4*(1-p)]/p

    return (4 * (1 - p) + 2 * p) / (p)


# Question 3
def q3(N):
    # Asserting that the teams must be even. According to our test statistic this must be true because each team
    # must play every day of the season.
    assert (N % 2 == 0)

    # Set up constants for iteration.
    seasons = iterations  # Determines number of iterations to run simulation
    pairs = N // 2  # Used to cut down on integer division later
    game_probabilities = [0.5] * pairs  # Sets probabilities to draw binomial distribution in loop

    # Set up space to store values
    teams = list(range(0, N))
    total_champion_wins = 0

    # Iterate over all "seasons" to be simulated
    for season in range(0, seasons):
        wins = [0] * N  # Reset wins vector each season

        # Iterate over all 16 games in season
        for day in range(0, 16):
            # Shuffle teams playing against each other for each day
            np.random.shuffle(teams)

            # Draw winner for games from Bernoulli Distribution: 1 = first team wins, 0 = second team wins
            winners = np.random.binomial(1, game_probabilities)

            # Iterate over each pair of teams (N/2 games per day for each team)
            for pair in range(0, pairs):
                if winners[pair] == 1:
                    wins[teams[pair * 2]] += 1  # Adds a win into the win vector for team 1 of the game
                else:
                    wins[teams[pair * 2 + 1]] += 1  # Adds a win into the win vector for team 2 of the game

        total_champion_wins += max(wins)  # Find the team with the max wins and add it to the total

    return total_champion_wins / seasons

## Unused function from misinterpreting homework statement:

# def findLongestSequence(array_in):
#     current_element = array_in[0]
#     longest_sequence = -1
#     current_sequence = 1
#
#     for i in range(1, len(array_in)):
#         if array_in[i] == current_element:
#             current_sequence += 1
#         else:
#             if current_sequence > longest_sequence:
#                 longest_sequence = current_sequence
#             current_sequence = 1
#             current_element = array_in[i]
#
#     if current_sequence > longest_sequence:
#         longest_sequence = current_sequence
#
#     return longest_sequence
