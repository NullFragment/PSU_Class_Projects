#include "stdafx.h"
#include "dijkstraNode.h"


DijkstraNode::DijkstraNode(int row, char name)
{
	rowValue = row;
	nameValue = name;
}

DijkstraNode::~DijkstraNode()
{
}

bool DijkstraNode::operator>(const DijkstraNode & node1) const
{
	return rowValue > node1.rowValue;
}

bool DijkstraNode::operator<(const DijkstraNode & node1) const
{
	return rowValue < node1.rowValue;
}