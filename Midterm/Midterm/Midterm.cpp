// Midterm.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

using namespace std;

int main()
{
	// -------------------------------------------------
	// BEGIN MIDTERM PRACTICE
	// -------------------------------------------------

	int hw1_i;
	const int hw1_total_columns = 2;
	const int hw1_total_rows = 4;
	struct mydata
	{
		int gate[hw1_total_columns];
	}

	circuit[hw1_total_rows] =
	{ 
		{ 1, 1 },
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 0 },

	};

	cout << "--------Midterm: Truth Table--------" << endl;

	for (hw1_i = 0; hw1_i < hw1_total_rows; hw1_i++)
	{
		if (
			((circuit[hw1_i].gate[0] && circuit[hw1_i].gate[1]) ||
			(circuit[hw1_i].gate[0] && !circuit[hw1_i].gate[1])) ||
			((circuit[hw1_i].gate[0] && circuit[hw1_i].gate[1]) ||
			(!circuit[hw1_i].gate[0] && circuit[hw1_i].gate[1]))
			)
		{
			cout << "True  - ON:  " << circuit[hw1_i].gate[0]
				<< " " << circuit[hw1_i].gate[1]
				<< " | 1" << endl;
		}
		else
		{
			cout << "False - OFF: " << circuit[hw1_i].gate[0]
				<< " " << circuit[hw1_i].gate[1]
				<< " | 0" << endl;
		}
	}
	// -------------------------------------------------
	// END Midterm
	// -------------------------------------------------


	cout << "\nPress enter to end." << endl;
	getchar();
	return 0;
}