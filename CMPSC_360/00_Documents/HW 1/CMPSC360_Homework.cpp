// CMPSC360_Homework.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
using namespace std;
int main()
{
	// -------------------------------------------------
	// BEGIN HOMEWORK 1
	// -------------------------------------------------
	int hw1_i;
	const int hw1_total_columns = 3;
	const int hw1_total_rows = 8;
	struct mydata
	{
		int gate[hw1_total_columns];
	}
	circuit[hw1_total_rows] = { { 1, 1, 1 },
				    { 1, 1, 0 },
				    { 1, 0, 1 },
				    { 1, 0, 0 },
				    { 0, 1, 1 },
				    { 0, 1, 0 },
				    { 0, 0, 1 },
				    { 0, 0, 0 } };

	cout << "--------Homework 1: Truth Table--------" << endl;
	for (hw1_i = 0; hw1_i < hw1_total_rows; hw1_i++)
	{
		if (circuit[hw1_i].gate[2] &&
			(!circuit[hw1_i].gate[0] || !circuit[hw1_i].gate[1]))
		{
			cout << "True  - ON:  " << circuit[hw1_i].gate[0]
				<< " " << circuit[hw1_i].gate[1]
				<< " " << circuit[hw1_i].gate[2]
				<< " | 1" << endl;
		}
		else
		{
			cout << "False - OFF: " << circuit[hw1_i].gate[0]
				<< " " << circuit[hw1_i].gate[1]
				<< " " << circuit[hw1_i].gate[2]
				<< " | 0" << endl;
		}
	}
	// -------------------------------------------------
	// END HOMEWORK 1
	// -------------------------------------------------
	cout << "\nPress enter to end." << endl;
	getchar();
	return 0;
}