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
		output << "New Shortest Distance:" << '\t' << length << '\t' << " Path: " << '\t';
	}
	else if (shortest == false)
	{
		output << "New Longest Distance:" << '\t' << length << '\t' << " Path: " << '\t';
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
	std::ofstream outfile;
	outfile.open("finaltask.txt");
	time_t current_Time = time(0);
	const int elements = 4;
	int a, b, cycles, loops = 0;
	std::vector<int> list;
	std::vector<std::vector<int>> permutations = {};
	std::vector<int> shortestDistance = {};
	std::vector<int> longestDistance = {};

	
	int weights[elements][elements] = 
	{
		{0, 432, 1436, 2509},
		{ 432,0,1779,2852 },
		{ 1436,1779,0,1317 },
		{ 2509,2852,1317,0 }
	};
	
	/*
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
	*/

	for (int i = 1; i <= elements; i++)
	{
		list.push_back(i);
	}

	std::random_shuffle(list.begin(), list.end());
	std::cout << "Choose a reasonable value (under 10,000). 10,000 will take approximately an hour to run." << std::endl  << "Enter Number of Paths to Find: ";
	std::cin >> cycles;
	
	std::cout << "Start Time: " << ctime(&current_Time);
	outfile << "Start Time: " << ctime(&current_Time);

	for (int i = 0; i < cycles; i++)
	{
		if ((i+1)% 100 == 0)
		{
			std::cout << "Cycle: " << i+1 << std::endl;
		}
		loops = 0;
		while (std::find(permutations.begin(), permutations.end(), list) != permutations.end() && loops <= 100)
		{
			std::random_shuffle(list.begin(), list.end());
			loops++;
		}
		permutations.push_back(list);
	}

	for each (std::vector<int> currentList in permutations)
	{
		int newDistance = 0;
		for (int i = 1; i < elements; i++)
		{
			a = currentList[i-1] - 1;
			b = currentList[i] - 1;
			newDistance = newDistance + weights[a][b];
		}
		
		if (shortestDistance.empty())
		{
			shortestDistance.push_back(newDistance);
			PrintPath(currentList, newDistance, true, outfile);
		}
		else if (newDistance < shortestDistance.back())
		{
			shortestDistance.push_back(newDistance);
			PrintPath(currentList, newDistance, true, outfile);
		}

		if (longestDistance.empty())
		{
			longestDistance.push_back(newDistance);
			PrintPath(currentList, newDistance, false, outfile);

		}
		else if (newDistance > longestDistance.back())
		{
			longestDistance.push_back(newDistance);
			PrintPath(currentList, newDistance, false, outfile);
		}
	}
	current_Time = time(0);
	outfile <<"End Time: " << ctime(&current_Time) << std::endl;
	std::cout << "End Time: " << ctime(&current_Time) << std::endl;
	system("pause");

	return 0;
}