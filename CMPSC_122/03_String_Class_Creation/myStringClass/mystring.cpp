/**
File Name: mystring.h

Name: Kyle Salitrik
PSU ID: 997543474
Due Date: 7/10/2013 | 11:55 PM
Last Modification: 7/9/2013 | 3:42 AM

Description:
This file includes function definitions for the string class functions.


Input: Strings, characters, character arrays, intergers.
Output: Intergers, characters, strings.
**/

#include <iostream>
#include <cstring>
#include "mystring.h"

/* Constructors */

String::String() // This function constructs an empty string.
{
	contents[0] = '\0';
	len = 0;
}

String::String(const char s[]) // This function constructs a string with the contents and length of array s.
{
	strcpy(contents, s);
	len = strlen(s);
}

/* Basic Functions */

void String::assign(const char s[]) // This function assigns the contents and length of array s to the called string.
{
	strcpy(contents, s);
	len = strlen(s);
}

void String::append(const String &str) // This function appends the passed string to the called string.
{
	strcat(contents, str.contents);
	len = strlen(contents);
}

int String::compare_to(const String &str) const // This string compares the called string to the passed string and returns a numerical value depending on the
												// result. The value is positive if the called string is greater, negative if it is lesser and zero if the
												// strings are the same.
{
	int result = strcmp(contents, str.contents);
	return result;
}	

void String::print() const // This function prints the contents of the string out to the display.
{
	cout << contents;
}

void String::print(ostream &out) const // Prints the contents of the string using a user-defined output.
{
	out << contents;
}

int String::length() const // This function returns the length of the string.
{
	return len;
}

char String::element(int n) // This function returns the n-th element of the string, or if the value is out of bounds, it returns an error message with
							// the length of the string.
{
	if ( n > len) 
	{
			cerr << "Element requested is outside of string boundaries. Please use a value lower than: " <<len;
			return '\0';
	}

	else return contents[n];
}

/* Operator Overloads */

void String::operator+=(const String &str) // This function works identically to the append function.
{
	strcat(contents, str.contents);
	len = strlen(contents);
}

char String::operator[](int n) const // This function works identically to the element function.
{
	if ( n > len) 
	{
			cerr << "Element requested is outside of string boundaries. Please use a value lower than: " <<len;
			return '\0';
	}

	else return contents[n];
}

bool String::operator==(const String &str) const // This function returns a true boolean value if the strings are identical, or false if they are not.
{
	if (strcmp == 0) return true;
	else return false;
}

bool String::operator!=(const String &str) const // This function returns a true boolean value if the strings are not identical and false if they are.
{
	if (strcmp == 0) return false;
	else return true;
}

bool String::operator<(const String &str) const // This function returns a true boolean value if the called string is less than the passed string.
{
	if (strcmp < 0) return true;
	else return false;
}

bool String::operator<=(const String &str) const // This function returns a true boolean value if the called string is less than or equal to the passed string.
{
	if (strcmp <= 0) return true;
	else return false;
}

bool String::operator>(const String &str) const // This function returns a true boolean value if the called string is greater than the passed string.
{
	if (strcmp > 0) return true;
	else return false;
}

bool String::operator>=(const String &str) const // This function returns a true boolean value if the called string is greater than or equal to the passed one.
{
	if (strcmp >= 0) return true;
	else return false;
}


/* External  Overloads */
ostream & operator<<(ostream &out, const String & r) // Overloads the << operator to externally access the user defined output print function of the string.
{
	r.print(out);
	return out;
}


/* OUTPUT FROM TEST PROGRAM

Enter a value for str1 (no spaces): Hello

Enter a value for str2 (no spaces): World

Enter a value for str3 (no spaces): Test
Initial values:
str1 holds "Hello" (length = 5)
str2 holds "World" (length = 5)
str3 holds "Test" (length = 4)

Enter which element of str1 to display (an integer for the subscription): 1
Element #1 of str1 is 'e'

Enter which element of str2 to display (an integer for the subscription): 0
Element #0 of str2 is 'W'

Enter which element of str3 to display (an integer for the subscription): 3
Element #3 of str3 is 't'

Append str2 to str1:
After appending str1 is:
str1 holds "HelloWorld" (length = 10)

Comparing str1 and str2...
"HelloWorld" is greater than "World"
 Type any non-return key to end the program:

 */