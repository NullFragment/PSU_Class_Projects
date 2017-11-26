#ifndef ASSIGNMENT_4_FUNCTIONS_RECORDS_H
#define ASSIGNMENT_4_FUNCTIONS_RECORDS_H

#include "utils.h"

void getIndexedRecord(_table *schema, linkedList *selects, FILE* output);

void getRecord(_table *schema, linkedList *selects, FILE* output);

bool selectRecord(char *buffer);

#endif //ASSIGNMENT_4_FUNCTIONS_RECORDS_H
