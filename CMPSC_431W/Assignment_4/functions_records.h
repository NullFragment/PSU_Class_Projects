#ifndef ASSIGNMENT_4_FUNCTIONS_RECORDS_H
#define ASSIGNMENT_4_FUNCTIONS_RECORDS_H

#include "utils.h"

void getRecord(struct _table *schema, char *fields, char *to_match, char *condition);

bool selectRecord(char *buffer);

#endif //ASSIGNMENT_4_FUNCTIONS_RECORDS_H
