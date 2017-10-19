#ifndef ASSIGNMENT_4_SCHEMA_FUNCTIONS_H
#define ASSIGNMENT_4_SCHEMA_FUNCTIONS_H

#include "utils.h"

bool loadSchema(_table *table, char *buffer);

bool createSchema(char *schema_name, char *buffer, FILE *stream, bool append, bool logging);

void printSchema(_table *schema);

void createTempSchema(char *first, char *second, char *temp_name);

#endif //ASSIGNMENT_4_SCHEMA_FUNCTIONS_H
