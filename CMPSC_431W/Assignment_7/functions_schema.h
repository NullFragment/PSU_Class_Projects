#ifndef ASSIGNMENT_4_SCHEMA_FUNCTIONS_H
#define ASSIGNMENT_4_SCHEMA_FUNCTIONS_H

#include "utils.h"

bool loadSchema(_table *table, char *buffer);

bool createSchema(char *schema_name, char *buffer, FILE *stream, bool append, bool logging);

// void printSchema(_table *schema);

void createTempSchema(char *first, char *second, char *temp_name);

void createIndex(char *buffer, FILE *stream);

void loadIndex(char *indexName, _table *baseTable, linkedList *indexOn, fieldList *idxFields);

void parseFile(FILE *toParse, FILE *output, fieldList *fields, bool comma);

#endif //ASSIGNMENT_4_SCHEMA_FUNCTIONS_H
