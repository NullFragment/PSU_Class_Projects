/**
File Name: mystring.h

Name: Kyle Salitrik
PSU ID: 997543474
Due Date: 7/3/2012 | 11:55 PM
Last Modification: 7/1/2012 | 2:22 PM

Description:
This header file defines the functions of MyStrLen, MyStrCompare, MyStrCopy and MyStrConcat using pointers in order to replicate basic string operations.

Input: Character strings
Output: Intergers, character strings.
**/


// The MyStrLen function uses a pointer directed to the memory location of an input string, and increments that location until the null character is found.
// once the null character is found it returns the amount of characters in that string.

int MyStrLen (const char *inputString)
{
	int stringLength = 0;
	const char *stringLocation = inputString;
	while (*stringLocation != '\0')
	{
		++stringLocation;
		++stringLength;
	}
	return stringLength;
}

// The MyStrCompare function uses pointers directed to the locations of two input strings and compares them to see if the lengths of the strings are equal
// then returns a value to show the result. 0 = equal length, 1 = string 1 is longer, -1 = string 2 is longer.

int MyStrCompare (const char *inputString1, const char *inputString2)
{
	const char *stringLocation1 = inputString1;
	const char *stringLocation2 = inputString2;
	int result = 0;
	while (*stringLocation1 != '\0' || *stringLocation2 != '\0')
	{
		if (*stringLocation1 == '\0')
		{
			--result;
			return result;
		}

		else if (*stringLocation2 == '\0')
		{
			++result;
			return result;
		}

		++stringLocation1;
		++stringLocation2;
	}
	return result;
}

// The MyStrCopy function takes an input of two strings, assigns pointers to the first memory locations and copies the values of the second string into
// the first, overwriting the original locations.

char* MyStrCopy(char *inputString1, const char *inputString2)
{
	char *stringOne = inputString1;

    while(*inputString2 != '\0')
	{
		*stringOne = *inputString2;
		stringOne++;
		inputString2++;
	}
	
	*stringOne = '\0';

	return inputString1;
}


//The MyStrCat function attaches string 2 to the end of string 1 using pointers assigned to input strings.

char* MyStrConcat(char *inputString1, const char *inputString2)
{
	char *stringOne = inputString1;
	
	while(*stringOne != '\0')
	{
		stringOne++;
	}

    while(*inputString2 != '\0')
	{
		*stringOne = *inputString2;
		stringOne++;
		inputString2++;
	}
	
	*stringOne = '\0';

	return inputString1;
}