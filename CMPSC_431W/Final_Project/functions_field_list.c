#include "functions_field_list.h"

// #############################################################################
// ### FIELD LIST FUNCTIONS
// #############################################################################

void fillFieldNode(fieldNode *to_fill, char *fieldName, char *fieldType, int length)
{
    to_fill->length = length;
    to_fill->fieldName = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->fieldName, fieldName, MAXINPUTLENGTH - 1);
    to_fill->fieldType = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->fieldType, fieldType, MAXINPUTLENGTH - 1);
}


fieldList *makeFieldList(char *fieldName, char *fieldType, int length)
{
    fieldList *list = (fieldList *) calloc(sizeof(fieldList), 1);
    list->count = 1;
    list->head = calloc(sizeof(fieldNode), 1);
    list->tail = list->head;
    fillFieldNode(list->head, fieldName, fieldType, length);
    return list;
}

bool addfieldNode(fieldList *list, bool at_head, char *fieldName, char *fieldType, int length)
{
    if (at_head == false && list->tail != NULL)
    {
        list->tail->next = calloc(sizeof(fieldNode), 1);
        list->tail = list->tail->next;
        fillFieldNode(list->tail, fieldName, fieldType, length);
        list->count++;
        return true;
    } else if (at_head == true && list->head != NULL)
    {
        fieldNode *temp = calloc(sizeof(fieldNode), 1);
        temp->next = list->head;
        list->head = temp;
        fillFieldNode(list->head, fieldName, fieldType, length);
        list->count++;
        return true;
    } else if (list->count == 0 && list->head == NULL)
    {
        list->head = calloc(sizeof(fieldNode), 1);
        fillFieldNode(list->head, fieldName, fieldType, length);
        list->count++;
        if (list->tail == NULL)
        {
            list->tail = list->head;
        }
        return true;
    } else return false;
}

void popFieldNode(fieldList *list)
{
    if (list->count > 0 && list->head != NULL)
    {
        fieldNode *temp = list->head;
        list->head = list->head->next;
        free(temp);
        list->count--;
    }
    if (list->count == 0)
    {
        list->tail = NULL;
    }
}