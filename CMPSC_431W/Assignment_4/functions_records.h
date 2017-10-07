#ifndef ASSIGNMENT_4_FUNCTIONS_RECORDS_H
#define ASSIGNMENT_4_FUNCTIONS_RECORDS_H

#include "utils.h"

bool getRecord(int recnum, char *record, struct _table *table);

void showRecord(struct _field *fields, char *record, int fieldcount);

void selectRecord(struct _table *schema, char *fields);

#endif //ASSIGNMENT_4_FUNCTIONS_RECORDS_H
