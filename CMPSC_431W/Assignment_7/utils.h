#ifndef ASSIGNMENT_4_UTILS_H
#define ASSIGNMENT_4_UTILS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#define MAXINPUTLENGTH 5000
#define MAXBINSEARCH 100

typedef enum
{
    false, true
} bool;

typedef struct node
{
    char *field;
    char *condition;
    bool constant;
    struct node *next;
} node;

typedef struct
{
    node *head;
    node *tail;
    int count;
} linkedList;

typedef struct fieldNode
{
    char *fieldName;
    char *fieldType;
    int length;
    struct fieldNode *next;
} fieldNode;

typedef struct
{
    fieldNode *head;
    fieldNode *tail;
    int count;
} fieldList;

typedef struct
{
    char *tableFileName;
    int reclen;
    int fieldcount;
    bool index;
    fieldList *fields;
} _table;

void trimwhitespace(char *to_trim);

char *trimChars(char *string, char *to_trim);

#endif //ASSIGNMENT_4_UTILS_H
