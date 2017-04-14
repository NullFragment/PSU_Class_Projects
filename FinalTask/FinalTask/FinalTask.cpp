// FinalTask.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include <vector>

int main()
{
	const int elements = 4;
	int a, b = 0;
	std::vector<int> list;
	std::vector<int> distances = {};

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

		if (distances.empty())
		{
			distances.push_back(newDistance);
			std::cout << "New Shortest Distance: " << newDistance << " Path: ";
			for (int i = 0; i < elements; i++)
			{
				if (i != 0)
				{
					std::cout << " -> ";
				}
				std::cout << list[i];
			}
			std::cout << '\n';
		}
		else if (newDistance < distances.back())
		{
			distances.push_back(newDistance);
			std::cout << "New Shortest Distance: " << newDistance << " Path: ";
			for (int i = 0; i < elements; i++)
			{
				if (i != 0)
				{
					std::cout << " -> ";
				}
				std::cout << list[i];
			}
			std::cout << '\n';
		}
	} while (std::next_permutation(list.begin(), list.end()));
	

	system("pause");

	return 0;
}


