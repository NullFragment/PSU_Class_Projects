#ifndef ASSIGNMENT_4_FUNCTIONS_DATABASE_H
#define ASSIGNMENT_4_FUNCTIONS_DATABASE_H

#include "utils.h"

bool dropTable(char *schema_name);

bool loadDatabase(_table *table, char *buffer);

#endif //ASSIGNMENT_4_FUNCTIONS_DATABASE_H
