#ifndef ASSIGNMENT_4_SCHEMA_FUNCTIONS_H
#define ASSIGNMENT_4_SCHEMA_FUNCTIONS_H

#include "utils.h"

bool loadSchema(struct _table *table, char *buffer);

bool createSchema(char *file_name, char *buffer);

void printSchema(struct _table *schema);

#endif //ASSIGNMENT_4_SCHEMA_FUNCTIONS_H
