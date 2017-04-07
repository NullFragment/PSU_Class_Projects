// Bounty2-Dijkstra.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "limits.h"
#include <iostream>
#include <map>
#include <queue>
#include <vector>
#include <functional>
#include "dijkstraNode.h"
#include "graph.h"


#define Nodes 7

int main()
{
	std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> heapStructure;
	heapStructure.emplace(1, 'a');
	heapStructure.emplace(2, 'b');

	Graph dijkstraGraph = Graph(7, 'a', 'g');
	dijkstraGraph.InitializeDistance();


	int gridSpace[Nodes][Nodes] = 
	{
		{ 0, 4, 3, 0, 7, 0, 0 },
		{ 4, 0, 6, 5, 0, 0, 0 },
		{ 3, 6, 0, 11, 8, 0, 0 },
		{ 0, 5, 11, 0, 2, 2, 10 },
		{ 7, 0, 8, 2, 0, 0, 5 },
		{ 0, 0, 0, 2, 0, 0, 3 },
		{ 0, 0, 0, 10, 5, 3, 0 }
	};



	system("pause");
    return 0;
}

