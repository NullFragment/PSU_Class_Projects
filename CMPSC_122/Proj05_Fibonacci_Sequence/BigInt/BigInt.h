//cmpsc122 Su 2012 Assignment 6 
//Modified from textbook Larry Nyhoff,  ADTs, Data Structures, and Problem Solving 
//with C++, 2nd ed., Prentice-Hall, 2005.

/*-- BigInt.h -------------------------------------------------------------
 
  This header file defines the data type BigInt for processing 
  integers of any size.
  Basic operations are:
     Constructor
     +:         Addition operator
     read():    Read a BigInt object
     display(): Display a BigInt object
     <<, >> :   Input and output operators
-------------------------------------------------------------------------*/

#include <iostream>
#include <iomanip>       // setfill(), setw()
#include <list>
#include <cmath>	// pow

#ifndef BIGINT
#define BIGINT

const int DIGITS_PER_BLOCK = 3;

const int MODULUS = (short int)pow(10.0, DIGITS_PER_BLOCK);

class BigInt
{
 public:
	/***** Constructors *****/
	BigInt()
	{ }
	/*-----------------------------------------------------------------------
	Default cConstructor
	Precondition: None
	Postcondition: list<short int>'s constructor was used to build
	this BigInt object.
	-----------------------------------------------------------------------*/
	BigInt(int n);
	/*-----------------------------------------------------------------------
	Construct BigInt equivalent of n.
	Precondition: n >= 0.
	Postcondition: This BigInt is the equivalent of integer n.
	-----------------------------------------------------------------------*/
	/******** Function Members ********/
	/***** Constructor *****/
	// Let the list<short int> constructor take care of this.

	/***** read *****/
	void read(istream & in);
	/*-----------------------------------------------------------------------
		Read a BigInt.

		Precondition:  istream in is open and contains blocks of nonnegative
			integers having at most DIGITS_PER_BLOCK digits per block.
		Postcondition: Blocks have been removed from in and added to myList.
	-----------------------------------------------------------------------*/

	/***** display *****/
	void display(ostream & out) const;
	/*-----------------------------------------------------------------------
		Display a BigInt.

		Precondition:  ostream out is open.
		Postcondition: The large integer represented by this BigInt object
			has been formatted with the usual comma separators and inserted
			into ostream out. 
	------------------------------------------------------------------------*/

	/***** addition operator *****/
	BigInt operator+(BigInt addend2);
	/*------------------------------------------------------------------------
		Add two BigInts.

		Precondition:  addend2 is the second addend.
		Postcondition: The BigInt representing the sum of the large integer
		represented by this BigInt object and addend2 is returned.
	------------------------------------------------------------------------*/

	/***** boolean operator *****/
	bool operator>(BigInt comparedbigint);

	/***** subtraction operator *****/
	BigInt operator-(BigInt subtrahend);

	/***** multiplication operator *****/
	BigInt operator*(BigInt multiplier);
	
	private:
	/*** Data Members ***/
	list<short int> myList;
};  // end of BigInt class declaration

//-- Definition of constructor
inline BigInt::BigInt(int n)
{
	do
	{
		myList.push_front(n % MODULUS);
		n /= MODULUS;
	}
	while (n > 0);
}
//------ Input and output operators
inline istream & operator>>(istream & in, BigInt & number)
{
  number.read(in);
  return in;
}

inline ostream & operator<<(ostream & out, const BigInt & number)
{
  number.display(out);
  return out;
}

#endif
