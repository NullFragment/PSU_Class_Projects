/**
 * cmpsc122 Assignment 3-2 Programming Problem: String - Phase 2 test file 
 * File Name: Assign3_2driver.cpp
 *
 * Description: This program demonstrates a basic String class that implements
 * operator overloading, etc..
 *
 */
 
#include <iostream>
#include "mystring.h"

using namespace std;


/**Function Prototypes
 *
 * Prints out the value and length of the String object passed to it.
 * overloaded ostream operator << is used in the definition.
 */
void PrintString(const char *label, const String &str);   


/*************************** Main Program **************************/

int main()
{
  char s1[100], s2[100], s3[100];              // Some character strings.

	cout << "\nEnter a value for str1 (no spaces): ";
	cin >> s1;
	
	cout << "\nEnter a value for str2 (no spaces): ";
	cin >> s2;
	
	cout << "\nEnter a value for str3 (no spaces): ";
	cin >> s3;
   
    String str1(s1), str2(s2), str3(s3);  // Some String objects. Using constructor for copy
  // Print out their initial values...
	cout << "Initial values:" << endl;
  PrintString("str1", str1);
  PrintString("str2", str2);
  PrintString("str3", str3);

  // Access some elements...

  int i;

  cout << "\nEnter which element of str1 to display (an integer for the subscription): ";
  cin >> i;
  cout << "Element #" << i << " of str1 is '" << str1[i]
       << "'" << endl;

  cout << "\nEnter which element of str2 to display (an integer for the subscription): ";
  cin >> i;
  cout << "Element #" << i << " of str2 is '" << str2[i]
       << "'" << endl;

  cout << "\nEnter which element of str3 to display (an integer for the subscription): ";
  cin >> i;
  cout << "Element #" << i << " of str3 is '" << str3[i]
       << "'" << endl;

  // Append strings...

  cout << "\nAppend str2 to str1: ";
 // str1.append(s1); // the cstring is converted to String object here by the constructor 
  str1 += str2;   // same as above 

  cout << "\nAfter appending str1 is: " << endl;
  PrintString("str1", str1);

  // Compare some strings...

  cout << "\nComparing str1 and str2..." << endl;

  cout << "\"";

  cout<< str1;   // test the overloading of ostream operator <<

  cout << "\" is ";

  if (str1 < str2) {      // test the overloading of comparison operator <
    cout << "less than";
  } else if (str1 > str2) {
    cout << "greater than";
  } else {
    cout << "equal to";
  }

  cout << " \"";
  cout << str2;
  cout << "\"" << endl;

  cout << " Type any non-return key to end the program: "<< endl;

  char q;

  cin >> q; 

  return 0;
}


/*********************** Function Definitions **********************/

void PrintString(const char *label,
                 const String &str)
{
  cout << label << " holds \"";
  cout << str;						//  << is overloaded									
  cout << "\" (length = " << str.length() << ")" << endl;
}
