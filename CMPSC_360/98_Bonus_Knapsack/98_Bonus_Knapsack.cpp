// 98_Bonus_Knapsack.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <iostream>


int Knapsack(int knapsack_capacity, std::vector<int> item_weights, std::vector<int> item_values)
{
	std::vector< std::vector<int> > knpsck;
	knpsck.resize(item_values.size()+1, std::vector<int>(knapsack_capacity+1, 0));
	int a, b = 0;
	for (int i = 0; i <= item_values.size(); i++)
	{
		for (int j = 0; j <= knapsack_capacity; j++)
		{
			if (i == 0 || j == 0)
			{
				knpsck[i][j] = 0;
			}
			else if (item_weights[i - 1] <= j)
			{
				a = item_values[i - 1] + knpsck[i - 1][j - item_weights[i - 1]];
				b = knpsck[i - 1][j];
				if(a < b)
				{
					knpsck[i][j] = b;
				}
				else if (a > b)
				{
					knpsck[i][j] = a;
				}
			}
			else
			{
				knpsck[i][j] = knpsck[i][j - 1];
			}
		}
	}

	return knpsck[item_values.size()][knapsack_capacity];
}




int main()
{
	std::vector<int> weights;
	std::vector<int> values;
	char ans = 'y';
	int temp, capacity;
	do 
	{
		std::cout << "Enter the item's weight: ";
		std::cin >> temp;
		weights.push_back(temp);

		std::cout << "Enter the item's value: ";
		std::cin >> temp;
		values.push_back(temp);
		std::cout << "Do you want to add another item? (y/n) ";
		std::cin >> ans;

	} while (ans == 'y' || ans == 'Y');

	std::cout << "Enter the maximum weight of the knapsack: ";
	std::cin >> capacity;

	if (weights.size()== values.size())
	{
		std::cout << "The maximum value is: " << Knapsack(capacity, weights, values) << std::endl;
	}

	system("pause");

    return 0;
}

