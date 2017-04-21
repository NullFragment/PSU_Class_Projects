/* 
// Name: Kyle Salitrik
// ID Number: 997543474
// Project 5
// Due: 7/20/2013, 11:55 PM
// Last Modificatons: 7/20/2013, 11:20
/
// Inputs: Strings, Intergers
// Outputs: Strings, Intergers
*/



//cmpsc122 Assignment 6 
//Modified from textbook Larry Nyhoff,  ADTs, Data Structures, and Problem Solving 
//with C++, 2nd ed., Prentice-Hall, 2005.

/*-- BigInt.cpp-------------------------------------------------------------
                This file implements BigInt member functions.
--------------------------------------------------------------------------*/

#include <iostream>
#include <cmath>
using namespace std;

#include "BigInt.h"

//--- Definition of read()
void BigInt::read(istream & in)
{
  static bool instruct = true;
  if (instruct)
  {
     cout << "Enter " << DIGITS_PER_BLOCK << "-digit blocks, separated by "
            "spaces.\nEnter a negative integer in last block to signal "
            "the end of input.\n\n";
    instruct = false;
  }
  short int block;
  const short int MAX_BLOCK = (short) pow(10.0, DIGITS_PER_BLOCK) - 1;
  for (;;)
  {
    in >> block;
    if (block < 0) return;

    if (block > MAX_BLOCK)
      cerr << "Illegal block -- " << block << " -- ignoring\n";
    else
      myList.push_back(block);
  }
}

//--- Definition of display()
void BigInt::display(ostream & out) const
{ 
   int blockCount = 0;
   const int BLOCKS_PER_LINE = 20;   // number of blocks to display per line

   for (list<short int>::const_iterator it = myList.begin(); ; )
   {
      out << setfill('0'); 
      if (blockCount == 0)
         out << setfill(' '); 
 
      if (it == myList.end())
         return;

      out << setw(3) << *it;
      blockCount++ ;

      it++;
      if (it != myList.end())
      {
         out << ',';
         if (blockCount > 0 && blockCount % BLOCKS_PER_LINE == 0)
            out << endl;
      }
   }
}

//--- Definition of operator+()
BigInt BigInt::operator+(BigInt addend2)
{
   BigInt sum;
   short int first,                  // a block of 1st addend (this object)
             second,                 // a block of 2nd addend (addend2)
             result,                 // a block in their sum
             carry = 0;              // the carry in adding two blocks

   list<short int>::reverse_iterator // to iterate right to left
      it1 = myList.rbegin(),         //   through 1st list, and
      it2 = addend2.myList.rbegin(); //   through 2nd list

   while (it1 != myList.rend() || it2 != addend2.myList.rend())
   {
      if (it1 != myList.rend())
      { 
         first = *it1;
         it1++ ;
      }
      else
         first = 0;
      if (it2 != addend2.myList.rend())
      {
         second = *it2;
         it2++ ;
      }
      else
         second = 0;

      short int temp = first + second + carry;
      result = temp % 1000;
      carry = temp / 1000;
      sum.myList.push_front(result);
   }

   if (carry > 0)
      sum.myList.push_front(carry);

   return sum;
}


//--- Definition of operator>()
bool BigInt::operator>(BigInt comparedbigint)
{
	int count1 = 0, count2 = 0;
    bool comparison = false;
   
   list<short int>::iterator // to iterate left to right
      it1 = myList.begin(),         //   through 1st list, and
      it2 = comparedbigint.myList.begin(), //   through 2nd list
      frontval1,
	  frontval2;
   
   while(const int *it1 = 0)
   {
	   it1++;
   }

   while(const int *it2 = 0)
   {
	   it2++;
   }
   frontval1 = it1;
   frontval2 = it2;

   while(it1 != myList.end() || it2 != comparedbigint.myList.end() )
   {

	   if(it1 != myList.end() )
	   {
		   count1++; 
		   it1++;
	   }
	   if(it2 != comparedbigint.myList.end() )
	   {
		   count2++; 
		   it2++;
	   }
	   
   }

   if(count1 > count2) comparison = true;
   if(count1 == count2)
   {
	   while(frontval1 != myList.end() || frontval2 != myList.end() )
	   {

		   if( *frontval1 > *frontval2)
		   {
				   comparison = true; 
				   break;
		   }

		   if( *frontval1 < *frontval2)
		   {
			   break;
		   }

		   if(*frontval1 == *frontval2)
		   {
			   frontval1++;
			   frontval2++;
		   }
	   }

   }

   return comparison;
}

// Definition of operator-()
BigInt BigInt::operator-(BigInt subtrahend)
{
	BigInt difference;
	short int first,                  // a block of 1st addend (this object)
		second,                 // a block of 2nd addend (addend2)
		result,                 // a block in their sum
		carry = 0;              // the carry in adding two blocks

	list<short int>::reverse_iterator // to iterate right to left
		it1 = myList.rbegin(),         //   through 1st list, and
		it2 = subtrahend.myList.rbegin(); //   through 2nd list
while (it1 != myList.rend() || it2 != subtrahend.myList.rend())
   {
      if (it1 != myList.rend())
      { 
         first = *it1;
         it1++ ;
      }
      else
         first = 0;
      if (it2 != subtrahend.myList.rend())
      {
         second = *it2;
         it2++ ;
      }
      else
         second = 0;

      short int temp = first - second + carry;
      result = temp % 1000;
      carry = temp / 1000;
      difference.myList.push_front(abs(result));
   }

   if (result < 0)
      difference = 0;

   return difference;
}

// Definition of operator*()
//--- Definition of operator+()
BigInt BigInt::operator*(BigInt multiplier)
{
   BigInt product;
   short int first,                  // a block of 1st addend (this object)
             second,                 // a block of 2nd addend (addend2)
             result,                 // a block in their sum
             carry = 0;              // the carry in adding two blocks

   list<short int>::reverse_iterator // to iterate right to left
      it1 = myList.rbegin(),         //   through 1st list, and
      it2 = multiplier.myList.rbegin(); //   through 2nd list

   while (it1 != myList.rend() || it2 != multiplier.myList.rend())
   {
      if (it1 != myList.rend())
      { 
         first = *it1;
         it1++ ;
      }
      else
         first = 0;
      if (it2 != multiplier.myList.rend())
      {
         second = *it2;
         it2++ ;
      }
      else
         second = 0;

      short int temp = (first+carry) * second;
      result = temp % 1000;
      carry = temp / 1000;
      product.myList.push_front(result);
   }

   if (carry > 0)
      product.myList.push_front(carry);

   return product;
}

/**

Execution Trace:

 Enter an integer n to calculate the nth Fibonacci number:12
377
Enter a big integer:
Enter 3-digit blocks, separated by spaces.
Enter a negative integer in last block to signal the end of input.

000 000 012-1
Enter another big integer:
000 000 008-1
The sum of
          0,000,012 +   0,000,008
is
          0,000,020

The bigger number of
          0,000,012
and
          0,000,008
is
          0,000,012

The subtraction of
          0,000,012 -   0,000,008
is
          0,000,004

BONUS part:
The multiplication of
          0,000,012 *   0,000,008
is
          0,000,096

Add more integers (Y or N)? Y
Enter a big integer:
111 222 333-3
Enter another big integer:
444 555 666-6
The sum of
        111,222,333 + 444,555,666
is
        555,777,999

The bigger number of
        111,222,333
and
        444,555,666
is
        444,555,666

The subtraction of
        111,222,333 - 444,555,666
is
          0

BONUS part:
The multiplication of
        111,222,333 * 444,555,666
is
        -588,013,170

Add more integers (Y or N)?

*/