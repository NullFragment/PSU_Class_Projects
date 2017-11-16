#ifndef ASSIGNMENT_4_FUNCTIONS_DATABASE_H
#define ASSIGNMENT_4_FUNCTIONS_DATABASE_H

#include "utils.h"

bool dropTable(char *schema_name);

bool loadDatabase(_table *table, char *buffer);

bool checkWhereLiteral(_table *schema, node *table, linkedList *clauses);

bool joinTable(_table *first, _table *second, linkedList *clauses, char *temp_name);

#endif //ASSIGNMENT_4_FUNCTIONS_DATABASE_H
