// FinalTask.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include <vector>


void PrintPath(std::vector<int> path, int length, bool shortest)
{
	int loop = 0;
	if (shortest == true)
	{
		std::cout << "New Shortest Distance: " << length << " Path: ";
	}
	else if (shortest == false)
	{
		std::cout << "New Longest Distance: " << length << " Path: ";
	}

	for each (int i in path)
	{
		if (loop != 0)
		{
			std::cout << " -> ";
		}
		std::cout << i;
		i++;
	}
	std::cout << '\n';
}




int main()
{
	const int elements = 4;
	int a, b = 0;
	std::vector<int> list;
	std::vector<int> shortestDistance = {};
	std::vector<int> longestDistance = {};

	int weights[elements][elements] = 
	{
		{0, 432, 1436, 2509},
		{ 432,0,1779,2852 },
		{ 1436,1779,0,1317 },
		{ 2509,2852,1317,0 }
	};


	for (int i = 1; i <= elements; i++)
	{
		list.push_back(i);
	}

	std::sort(list.begin(), list.end());

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
			PrintPath(list, newDistance, true);
		}
		else if (newDistance < shortestDistance.back())
		{
			shortestDistance.push_back(newDistance);
			PrintPath(list, newDistance, true);
		}

		if (longestDistance.empty())
		{
			longestDistance.push_back(newDistance);
			PrintPath(list, newDistance, false);

		}
		else if (newDistance > longestDistance.back())
		{
			longestDistance.push_back(newDistance);
			PrintPath(list, newDistance, false);
		}
	} while (std::next_permutation(list.begin(), list.end()));
	

	system("pause");

	return 0;
}


