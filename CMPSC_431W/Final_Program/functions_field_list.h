#ifndef ASSIGNMENT_5_FIELD_LIST_H
#define ASSIGNMENT_5_FIELD_LIST_H

#include "utils.h"

void fillFieldNode(fieldNode *to_fill, char *fieldName, char *fieldType, int length);

fieldList *makeFieldList(char *fieldName, char *fieldType, int length);

bool addfieldNode(fieldList *list, bool at_head, char *fieldName, char *fieldType, int length);

void popFieldNode(fieldList *list);

#endif //ASSIGNMENT_5_FIELD_LIST_H
