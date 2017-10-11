#ifndef ASSIGNMENT_4_UTILS_H
#define ASSIGNMENT_4_UTILS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#define MAXFIELDS 100 // for now
#define MAXINPUTLENGTH 5000
#define MAXLENOFFIELDNAMES 50
#define MAXLENOFFIELDTYPES 50

struct _field
{
    char fieldName[MAXLENOFFIELDNAMES];
    char fieldType[MAXLENOFFIELDTYPES];
    int fieldLength;
};

struct _table
{
    char *tableFileName;
    int reclen;
    int fieldcount;
    struct _field fields[MAXFIELDS];
};

typedef enum
{
    false, true
} bool;

void trimwhitespace(char *to_trim);

char* trimQuotes(char *to_trim);

#endif //ASSIGNMENT_4_UTILS_H
