// FinalTask.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>



void PrintPath(std::vector<int> path, int length, bool shortest, std::ofstream &output)
{
	int loop = 0;
	if (shortest == true)
	{
		output << "New Shortest Distance: " << length << " Path: ";
	}
	else if (shortest == false)
	{
		output << "New Longest Distance: " << length << " Path: ";
	}

	for each (int i in path)
	{
		if (loop != 0)
		{
			output << " -> ";
		}
		output << i;
		loop++;
	}
	output << std::endl;
}




int main()
{
	std::ofstream permutations;
	permutations.open("finaltask.txt");
	time_t current_Time = time(0);
	const int elements = 14;
	int a, b = 0;
	std::vector<int> list;
	std::vector<int> shortestDistance = {};
	std::vector<int> longestDistance = {};

	/*
	int weights[elements][elements] = 
	{
		{0, 432, 1436, 2509},
		{ 432,0,1779,2852 },
		{ 1436,1779,0,1317 },
		{ 2509,2852,1317,0 }
	};
	*/
	
	int weights[elements][elements] =
	{
		{0, 13, 12, 3, 6, 5, 5, 5, 3, 1, 12, 7, 7, 3},
		{ 13,0,6,12,6,6,10,14,11,9,10,4,11,9 },
		{ 12,6,0,3,2,2,6,6,5,3,6,2,5,3 },
		{ 3,12,3,0,6,2,4,5,3,1,11,5,4,4 },
		{ 6,6,2,6,0,2,7,8,7,4,6,1,6,5 },
		{ 5,6,2,2,2,0,4,5,5,3,7,2,4,2 },
		{ 5,10,6,4,7,4,0,2,4,3,13,7,5,4 },
		{ 5,14,6,5,8,5,2,0,5,3,14,8,5,3 },
		{ 3,11,5,3,7,5,4,5,0,1,13,9,7,4 },
		{ 1,9,3,1,4,3,3,3,1,0,10,6,4,1 },
		{ 12,10,6,11,6,7,13,14,13,6,0,8,12,11 },
		{ 7,4,2,5,1,2,7,8,9,4,8,0,7,5 },
		{ 7,11,5,4,6,4,5,5,7,1,12,7,0,3 },
		{ 3,9,3,4,5,2,4,3,4,2,11,5,3,0 }

	};
	

	for (int i = 1; i <= elements; i++)
	{
		list.push_back(i);
	}

	std::sort(list.begin(), list.end());

	std::cout << "Start Time: " << ctime(&current_Time);
	permutations << "Start Time: " << ctime(&current_Time);
	do
	{
		int newDistance = 0;
		for (int i = 1; i < elements; i++)
		{
			a = list[i-1] - 1;
			b = list[i] - 1;
			newDistance = newDistance + weights[a][b];
		}

		if (shortestDistance.empty())
		{
			shortestDistance.push_back(newDistance);
			PrintPath(list, newDistance, true, permutations);
		}
		else if (newDistance < shortestDistance.back())
		{
			shortestDistance.push_back(newDistance);
			PrintPath(list, newDistance, true, permutations);
		}

		if (longestDistance.empty())
		{
			longestDistance.push_back(newDistance);
			PrintPath(list, newDistance, false, permutations);

		}
		else if (newDistance > longestDistance.back())
		{
			longestDistance.push_back(newDistance);
			PrintPath(list, newDistance, false, permutations);
		}
	} while (std::next_permutation(list.begin(), list.end()));
	current_Time = time(0);
	std::cout << "End Time: " << ctime(&current_Time) << std::endl;
	system("pause");

	return 0;
}


