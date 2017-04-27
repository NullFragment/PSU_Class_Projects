#include "stdafx.h"
#include "graph.h"

Graph::Graph(int vertices, char startNode, char endNode)
{
	Graph::vertices = vertices;
	Graph::lastNode = endNode;
	Graph::firstNode = startNode;
	std::map<char, int> nodePath, nodeDistance, heapMap;
}

Graph::~Graph()
{
}

void Graph::AddEdge(char start, char end, int weight)
{
}

void Graph::Dijkstra(int start)
{
}

void Graph::InitializeDistance()
{
	for (char node = 'a'; node <= lastNode; node++)
		nodeDistance.insert(std::pair<char, int>(node, INT_MAX));
	nodeDistance[firstNode] = 0;

	for (char node = 'a'; node <= lastNode; node++)
		std::cout << node << ": " << nodeDistance[node] << std::endl;
}
