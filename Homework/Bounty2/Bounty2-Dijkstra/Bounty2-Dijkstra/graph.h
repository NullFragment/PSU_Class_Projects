#pragma once

#include <list>
#include <utility>
#include <map>
#include "limits.h"
#include <iostream>

class Graph
{
public:
	Graph::Graph(int vertices, char startNode, char endNode);
	~Graph();
	void InitializeDistance();
	void AddEdge(char start, char end, int weight);
	void Dijkstra(int start);
private:
	int vertices;
	char firstNode;
	char lastNode;
	std::map<char, int> nodePath, nodeDistance, heapMap;
};