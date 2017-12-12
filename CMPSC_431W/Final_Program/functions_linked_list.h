#ifndef ASSIGNMENT_5_FUNCTIONS_LINKED_LIST_H
#define ASSIGNMENT_5_FUNCTIONS_LINKED_LIST_H

#include "utils.h"

void fillNode(node *to_fill, char *field, char *compareVal, int conditional, bool constant);

linkedList *makeLinkedList(char *field, char *compareVal, int conditional, bool constant);

bool addNode(linkedList *list, bool at_head, char *field, char *compareVal, int condition, bool constant);

void popNode(linkedList *list);

#endif //ASSIGNMENT_5_FUNCTIONS_LINKED_LIST_H
