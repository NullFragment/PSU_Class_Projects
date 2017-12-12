#include "functions_linked_list.h"

// #############################################################################
// ### LINKED LIST FUNCTIONS
// #############################################################################

void fillNode(node *to_fill, char *field, char *compareVal, int conditional, bool constant)
{
    to_fill->compareVal = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->compareVal, compareVal, MAXINPUTLENGTH - 1);
    to_fill->field = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->field, field, MAXINPUTLENGTH - 1);
    to_fill->constant = constant;
    to_fill->conditional = conditional;
}


linkedList *makeLinkedList(char *field, char *compareVal, int conditional, bool constant)
{
    linkedList *list = (linkedList *) calloc(sizeof(linkedList), 1);
    list->count = 1;
    list->head = calloc(sizeof(node), 1);
    list->tail = list->head;
    fillNode(list->head, field, compareVal, conditional, constant);
    return list;
}

bool addNode(linkedList *list, bool at_head, char *field, char *compareVal, int condition, bool constant)
{
    if (at_head == false && list->tail != NULL)
    {
        list->tail->next = calloc(sizeof(node), 1);
        list->tail = list->tail->next;
        fillNode(list->tail, field, compareVal, condition, constant);
        list->count++;
        return true;
    }
    else if (at_head == true && list->head != NULL)
    {
        node *temp = calloc(sizeof(node), 1);
        temp->next = list->head;
        list->head = temp;
        fillNode(list->head, field, compareVal, condition, constant);
        list->count++;
        return true;
    }
    else if (list->count == 0 && list->head == NULL)
    {
        list->head = calloc(sizeof(node), 1);
        fillNode(list->head, field, compareVal, condition, constant);
        list->count++;
        if (list->tail == NULL)
        {
            list->tail = list->head;
        }
        return true;
    }
    else return false;
}

void popNode(linkedList *list)
{
    if (list->count > 0 && list->head != NULL)
    {
        node *temp = list->head;
        list->head = list->head->next;
        free(temp);
        list->count--;
    }
    if (list->count == 0)
    {
        list->tail = NULL;
    }
}