/**
File Name: mystring.cpp

Name: Kyle Salitrik
PSU ID: 997543474
Due Date: 7/3/2012 | 11:55 PM
Last Modification: 7/1/2012 | 2:22 PM

Description:
This program tests the mystring.h header file functions of MyStrCompare, MyStrCopy, MyStrConcat and MyStrLen defined below as well as in the header then
prompts the user for a character input in order to pause before closing.

Input: Character strings
Output: Intergers, character strings.
**/

#include <cstdlib>
#include <iostream>
#include "mystring.h"

using namespace std;

int MyStrLen(const char *inputString); //  This function finds and returns the length of a string.
int MyStrCompare (const char *inputString1, const char *inputString2); // This function compares if two strings are of equal length. If yes, it
																	   // returns a value of 0, if the first string is longer it returns a value of 1
																	   // if the second string is longer it returns a value of -1.
char* MyStrCopy(char *inputString1, const char *inputString2); // This function copies the contents of the second input string into the first, provided
															   // there is enough memory allocated to do so.
char* MyStrConcat(char *inputString1, const char *inputString2); // This function concatenates two strings by attaching the second string onto the first
																 // provided there is enough memory allocated to do so.

int main()
{
	int result = 999;
	char* stringResult;
	char testString1[256] = "I";
	char endChar;

	//MyStrLen Tests
	result = MyStrLen ("");
	cout <<result <<endl;

	result = MyStrLen ("TestTest");
	cout <<result <<endl;

	result = MyStrLen ("TestTestTest");
	cout <<result <<endl;

	//MyStrCompare Tests
	result = MyStrCompare("Test1", "Test2");
	cout <<result <<endl;

	result = MyStrCompare("Test1", "Tes");
	cout <<result <<endl;

	result = MyStrCompare("Tes", "Test2");
	cout <<result <<endl;
	result = MyStrCompare("Tes", "");
	cout <<result <<endl;

	//MyStrCopy Tests
	cout <<"Starting string:   " <<testString1 <<endl;
	stringResult = MyStrCopy(testString1, "");
	cout <<"Blank string copy:   " <<stringResult <<endl;
	stringResult = MyStrCopy(testString1, "It");
	cout <<"Copying 'It':   " <<stringResult <<endl;

	//MyStrConcat Tests
	stringResult = MyStrConcat(testString1, "");
	cout <<"Empty string concatenate:   "<<stringResult <<endl;
	stringResult = MyStrConcat(testString1, " w");
	cout <<"Space and character concatenate:   "<<stringResult <<endl;
	stringResult = MyStrConcat(testString1, "orks.");
	cout <<"Multicharacter concatenate:   " <<stringResult <<endl;

	//Pause before closing.
	cout <<endl <<endl <<"Please input any character to close." <<endl;
	cin >> endChar;

	return 0;
}