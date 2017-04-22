// Kyle Salitrik
// kps168
// PSU ID: 997543474


#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;


// Functions

bool intSort(int i, int j) // Used in sort(), sorts intergers smallest to largest
{
	return i < j;
}

bool quiz2(bool P, bool Q) // This is a representation of a not(P and Q)
{
	

	if (P == false || Q == false)
	{
		// Prints Entered loop if not-P or not-Q exist.
		cout << "Entered Loop \n";
		return true;
	}
	else
	{
		// Prints loop bypassed if both P and Q are true.
		cout << "Loop bypassed \n";
		return false;
	}
}

bool quiz3(vector<int> &A, vector<int> &B, vector<int> &C) // Representation of the set A - BuC
{
	// This takes 3 interger arrays as inputs: A, B, C.
	// It then converts B and C to vectors and appends B to C.
	// Next, it iterates through A and subsequently the Union vector and
	// appends values that do not appear in B and C to a new  print vector.
	// Finally the print vector is output to the command line.

	bool pushon = true;												// Initalize print flag.
	std::vector<int> print = {};									// Initalize print vector.
	std::vector<int> unionBC = {};									// Initialize union vector.

	unionBC.insert(unionBC.end(), B.begin(), B.end());				// Append vector B to union vector.
	unionBC.insert(unionBC.end(), C.begin(), C.end());				// Append vector C to union vector.
	cout << "Union Vector: ";
	for (auto i = unionBC.begin(); i != unionBC.end(); ++i)			// Prints unionAB to console.
	{
		cout << *i << ' ';
	}
	cout << "\n";

	for (auto i = A.begin(); i != A.end(); ++i)						// Begin loop iteration through A.
	{
		for (auto j = unionBC.begin(); j != unionBC.end(); ++j)		// Begin loop iteration through unionAB.
		{
			if (*i == *j)											// If value of A exists in unionAB...
			{
				pushon = false;										// ...flag to not append to print vector.
			}
		}
		if (pushon == true)											// If value was not de-flagged...
		{
			print.push_back(*i);									// ...append value A[x] to print vector.
		}
		pushon = true;												// Reset print flag.
	}
	sort(print.begin(), print.end(), intSort);						// Sort print vector numerically
	cout << "Set of A - (BuC): ";
	for (auto i = print.begin(); i != print.end(); ++i)				// Loops through print vector
	{
		cout << *i << ' ';											// Sends element of print vector to console.
	}
	cout << "\n";													// Send newline to console.
	return true;
}

int main()
{
	std::vector<int> A = { 1,2,3,4, 27, 36, 9, 7 };
	std::vector<int> B = { 32, 6, 8, 10 };
	std::vector<int> C = { 1, 3, 5, 7, 9, 11 };

	bool P = true;
	bool Q = false;
	
	cout << "Quiz 2: P = " << P << "Q = " << Q << "\n";
	quiz2(P, Q);
	cout << "\n---------------------------------------\n\n";

	cout << "Quiz 3: \n";
	quiz3(A, B, C);
	cout << "\n---------------------------------------\n\n";

	cout << "Press return to end";
	cin.get();
	return 0;
}