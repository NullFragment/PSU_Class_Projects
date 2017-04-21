/**
File Name: mystring.h

Name: Kyle Salitrik
PSU ID: 997543474
Due Date: 7/10/2013 | 11:55 PM
Last Modification: 7/9/2013 | 3:42 AM

Description:
This file includes the class definition for the String class including its function prototypes.


Input: Strings, characters, character arrays, intergers.
Output: Intergers, characters, strings.
**/

# ifndef _MYSTRING_H
# define _MYSTRING_H

#include <iostream>
#include <cstring>

using namespace std;

#define MAX_STRING_LENGTH 256

class String
{
public:
	/* Constructors */
	String(); // Creates a blank string class.
	String(const char s[]); // Creates a string class initalized with the given character array.

	/* Operations */
	void assign(const char s[]); // Assigns the given array of characters to the string.
	void append(const String &str); // Appends a given string to the string being operated upon.
	int compare_to(const String &str) const; // Compares a given string to the inital string called.
	void print() const; // Prints the contents of the string.
	void print(ostream &out) const; // Prints the contents of the string using a user-defined output.
	int length() const; // Returns the length of the string.
	char element(int n); // Returns the n-th element of a string. If the n-th element is out of bounds an error message is sent to cerr with the string length.

	/* Operator Overloads */
	void operator +=(const String &str); // Overload function that acts like the append function.
	char operator [](int i) const; // This function behaves in the same fashion as the element function.
	bool operator ==(const String &str) const; // This function returns a true boolean value if the strings are equal.
	bool operator !=(const String &str) const; // This function returns a true boolean value if the strings are not equal.
	bool operator >(const String &str) const; // This function returns a true boolean value if the called string is greater than the passed string.
	bool operator <(const String &str) const; // This function returns a true boolean value if the called string is less than the passed string.
	bool operator >=(const String &str) const; // This function returns a true boolean value if the called string is greater than or equal to the passed one.
	bool operator <=(const String &str) const; // This function returns a true boolean value if the called string is less than or equal to the passed string.

private:
	char contents[MAX_STRING_LENGTH+1]; // This position stores the contents of the string as a charater array.
	int len; // This variable stores the length of the string.
};

#endif 

/* External Overloads */

ostream & operator<<(ostream &out, const String & r); // Overloads the << operator to externally access the user defined output print function of the string