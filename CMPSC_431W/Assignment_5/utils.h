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

typedef struct
{
    char fieldName[MAXLENOFFIELDNAMES];
    char fieldType[MAXLENOFFIELDTYPES];
    int fieldLength;
} _field;

typedef struct
{
    char *tableFileName;
    int reclen;
    int fieldcount;
    _field fields[MAXFIELDS];
} _table;

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



void trimwhitespace(char *to_trim);

char *trimQuotes(char *to_trim);

void fillNode(node *to_fill, char *field, char *condition, bool constant);

linkedList *makeLinkedList(char *field, char *condition, bool constant);

bool addNode(linkedList *list, bool at_head, char *field, char *condition, bool constant);

void popNode(linkedList *list);

#endif //ASSIGNMENT_4_UTILS_H
