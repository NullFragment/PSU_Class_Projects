/**
File Name: proj1.cpp

Name: Kyle Salitrik
PSU ID: 997543474
Due Date: 6/30/2012 | 11:55 PM
Last Modification: 6/26/2012 | 12:50 PM

Description:
This program prompts the user to start a number guessing game. If started, the program randomly generates a number between 1 and a set cap,
then proceeds to ask the user to guess the number. If the guess is too high or low, the program will say so until the correct number is guessed.
After winning, the user will be prompted to quit or begin a new game.

Input: Intergers and Characters into the command console.
Output: Strings.
**/

#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;



int main()
{
	// Initialization
	const int cap = 100;
	int randNum;
	int guess;
	char choice;
	
	
	srand(unsigned((time(NULL))));

	// Main Loop

	do
	{
		cout << "Would you like to (s)tart, watch the (c)omputer play or (q)uit?" << endl;
		cin >> choice;

		// Beginning of game code
		
		randNum = (rand() % cap + 1); // Generates a new number each time you start a new game as opposed to every time the game is opened.

		if (choice == 's')
		{
			
			do
			{
				cout << endl << "Please guess a number between 1 and " << cap << endl;
				cin >> guess;
				if (guess < randNum) cout << endl << "Your guess is too low." << endl;
				if (guess > randNum) cout << endl << "Your  guess is too high." << endl;
				if (guess == randNum) cout << endl << "You win!" << endl;
			}
			while (guess != randNum); // Break from loop when the user wins.
		}

		else if (choice == 'c') // Begin 'AI' round.
		{
			do
			{
				cout << endl << "Please guess a number between 1 and " << cap << endl;
				int compGuess = (rand() % cap + 1); // Generation of computer's guess.
				cout << compGuess; // Print out of the guess in order to show the number generated.
				guess = compGuess; // Assignment of the computer's number in order to continue the program.
				if (guess < randNum) cout << endl << "Your guess is too low." << endl;
				if (guess > randNum) cout << endl << "Your  guess is too high." << endl;
				if (guess == randNum) cout << endl << "You win!" << endl;
			}
			while (guess != randNum); // Break from loop when the user wins.
		}
	}

	while (choice != 'q'); // Break from prompt to start a new game.

	return 0;

}
