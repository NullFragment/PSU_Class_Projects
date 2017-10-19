#include "utils.h"

// #############################################################################
// ### UTILITY FUNCTIONS
// #############################################################################

/**
 * @brief Trims whitespace from a given character array
 * @param to_trim - pointer to array to trim whitespace from
 */
void trimwhitespace(char *to_trim)
{
    char *j;
    while (isspace(*to_trim))
    {
        to_trim++;
    }
    size_t length = strlen(to_trim);
    j = to_trim + length - 1;
    while (isspace(*j))
    {
        *j = 0;
        j--;
    }
}

/**
 * @brief Trims quotes from a given character array
 * @param to_trim - pointer to array to trim whitespace from
 */
char *trimQuotes(char *to_trim)
{
    char *j;
    while (strncmp(to_trim, "\"", 1) == 0)
    {
        to_trim++;
    }
    size_t length = strlen(to_trim);
    j = to_trim + length - 1;
    while (strcmp(j, "\"") == 0)
    {
        *j = 0;
        j--;
    }
    return to_trim;
}

void fillNode(node *to_fill, char *field, char *condition, bool constant)
{
    to_fill->condition = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->condition, condition, MAXINPUTLENGTH - 1);
    to_fill->field = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->field, field, MAXINPUTLENGTH - 1);
    to_fill->constant = constant;
}


linkedList *makeLinkedList(char *field, char *condition, bool constant)
{
    linkedList *list = (linkedList *) calloc(sizeof(linkedList), 1);
    list->count = 1;
    list->head = calloc(sizeof(node), 1);
    list->tail = list->head;
    fillNode(list->head, field, condition, constant);
    return list;
}

bool addNode(linkedList *list, bool at_head, char *field, char *condition, bool constant)
{
    if (at_head == false && list->tail != NULL)
    {
        list->tail->next = calloc(sizeof(node), 1);
        list->tail = list->tail->next;
        fillNode(list->tail, field, condition, constant);
        list->count++;
        return true;
    }
    else if (at_head == true && list->head != NULL)
    {
        node *temp = calloc(sizeof(node), 1);
        temp->next = list->head;
        list->head = temp;
        fillNode(list->head, field, condition, constant);
        list->count++;
        return true;
    }

    else if (list->count == 0 && list->head == NULL)
    {
        list->head = calloc(sizeof(node), 1);
        fillNode(list->head, field, condition, constant);
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