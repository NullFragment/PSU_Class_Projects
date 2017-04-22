/**
 * cmpsc122 Assignment 4 test file 
 * File Name: Assign4driver.cpp
 *
 * Description: This program demonstrates a basic String class that implements
 * dynamic allocation and operator overloading.
 *
 */
 
#include <iostream>
#include "mystring.h"
using namespace std;


/************************ Function Prototypes ************************/

/*
 * Function: PrintString
 * Usage: PrintString(str);
 *
 * Prints out the value and length of the String object passed to it.
 */
void PrintString(const char *label,
                 const String &str);    // overloaded ostream operator << is used in the definition.


/*************************** Main Program **************************/

int main()
{
  String str1, str2("init2"), str3 = "init3";  // Some String objects. Using constructor for copy
  char s1[100], s2[100], s3[100];              // Some character strings.

  // Print out their initial values...

  cout << "Initial values:" << endl;
  PrintString("str1", str1);
  PrintString("str2", str2);
  PrintString("str3", str3);

  // Store some values in them...

  cout << "\nEnter a value for str1 (no spaces): ";
  cin >> s1;
  str1 = s1;

  cout << "\nEnter a value for str2 (no spaces): ";
  cin >> s2;
  str2 = s2;

  cout << "\nEnter a value for str3 (no spaces): ";
  cin >> s3;
  str3 = s3;
 

  cout << "\nAfter assignments..." << endl;
  PrintString("str1", str1);
  PrintString("str2", str2);
  PrintString("str3", str3);

  // Access some elements...

  int i;

  cout << "\nEnter which element of str1 to display: ";
  cin >> i;
  cout << "Element #" << i << " of str1 is '" << str1[i]
       << "'" << endl;

  cout << "\nEnter which element of str2 to display: ";
  cin >> i;
  cout << "Element #" << i << " of str2 is '" << str2[i]
       << "'" << endl;

  cout << "\nEnter which element of str3 to display: ";
  cin >> i;
  cout << "Element #" << i << " of str3 is '" << str3[i]
       << "'" << endl;

  // Append some strings...

  cout << "\nEnter a value to append to str1 (no spaces): ";
  cin >> s1;
 // str1.append(s1); // Actually, the cstring is converted to String object here by the constructor 
  str1 += s1;   // same as above 

  cout << "\nEnter a value to append to str2 (no spaces): ";
  cin >> s2;
  str2 += s2;

  
  cout << "\nEnter a value to append to str3 (no spaces): ";
  cin >> s3;
  str3 += s3;

  cout << "\nAfter appending..." << endl;
  PrintString("str1", str1);
  PrintString("str2", str2);
  PrintString("str3", str3);

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

  cout << "\ntest the = operator, after str1 = str2; "<< endl;

  str1 = str2;

  PrintString("str1", str1);
  PrintString("str2", str2);

  str1 += s1;

  cout << "\nAfter str1 = str1 + s1: "<< endl;

  PrintString("str1", str1);
  PrintString("str2", str2);

  String str4(str3);
  cout << "\ntest the copy constructor, after str4(str3);"<< endl;
   
  PrintString("str3", str3);
  PrintString("str4", str4);

  cout << "\nafter appending str3 by str2" << endl;
  str3 += str2;
  PrintString("str3", str3);
  PrintString("str4", str4);

  cout<< "\nstr2, str4 are not changed. Type any letter to quit." << endl; 

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
