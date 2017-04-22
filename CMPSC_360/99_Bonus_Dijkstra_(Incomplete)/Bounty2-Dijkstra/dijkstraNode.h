#pragma once

class DijkstraNode
{
public:
	DijkstraNode(int row, char name);
	~DijkstraNode();
	int rowValue;
	char nameValue;
	bool operator > (const DijkstraNode &node1) const;
	bool operator < (const DijkstraNode &node1) const;
};
