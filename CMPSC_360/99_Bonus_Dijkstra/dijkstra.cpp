#include "stdafx.h"
#include <limits.h>
#include <vector>
#include <iostream>
#include <map>
#include "dijkstra.h"

int nextNode(std::vector<int> distances, std::vector<bool> visitedNodes)
{
	int currentPath = INT_MAX;
	int shortestPathIndex = 0;

	for (int node = 0; node < distances.size(); node++)
	{
		if (visitedNodes[node] == false && distances[node] <= currentPath)
		{
			currentPath = distances[node];
			shortestPathIndex = node;
		}
	}

	return shortestPathIndex;
}

void PrintPaths(std::vector<int> distances, std::map<int, char> nodeNames)
{
	std::cout << "Vertex" << '\t' << "Distance" << std::endl;
	for (int i = 0; i < distances.size(); i++)
	{
		std::cout << nodeNames[i] << '\t' << distances[i] << std::endl;
	}
}

void dijkstraAlgorithm(std::vector< std::vector<int>> graph, int sourceNode, std::map<int, char> nodeNames)
{
	std::vector<int> distances(graph.at(1).size(), INT_MAX);
	std::vector<bool> visitedNodes(graph.at(1).size(), false);

	distances[sourceNode] = 0;

	int newDistance, weight;

	for (int node = 0; node < 7; node++)
	{
		int from = nextNode(distances, visitedNodes);
		visitedNodes[from] = true;

		for (int to = 0; to < 7; to++)
		{
			weight = graph[from][to];
			newDistance = distances[from] + weight;
			if (visitedNodes[to] == false && weight > 0 && newDistance < distances[to] && distances[from] < INT_MAX)
			{
				distances[to] = distances[from] + graph[from][to];
			}
		}
	}
	PrintPaths(distances, nodeNames);
}