// 99_Bonus_Dijkstra.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <map>
#include "dijkstra.h"

int main()
{
	std::vector< std::vector<int> > gridSpace =
	{
		{ 0, 4, 3, 0, 7, 0, 0 },
		{ 4, 0, 6, 5, 0, 0, 0 },
		{ 3, 6, 0, 11, 8, 0, 0 },
		{ 0, 5, 11, 0, 2, 2, 10 },
		{ 7, 0, 8, 2, 0, 0, 5 },
		{ 0, 0, 0, 2, 0, 0, 3 },
		{ 0, 0, 0, 10, 5, 3, 0 }
	};

	std::map<int, char> nodeNames;
	nodeNames[0] = 'A';
	nodeNames[1] = 'B';
	nodeNames[2] = 'C';
	nodeNames[3] = 'D';
	nodeNames[4] = 'E';
	nodeNames[5] = 'F';
	nodeNames[6] = 'G';

	dijkstraAlgorithm(gridSpace, 0, nodeNames);

	system("Pause");

    return 0;
}