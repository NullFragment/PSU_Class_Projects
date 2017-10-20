#include "utils.h"
#include "functions_records.h"
#include "functions_schema.h"
#include "functions_linked_list.h"

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################


/**
 * @brief
 * @param schema - l
 * @param selects
 */
void getRecord(_table *schema, linkedList *selects)
{
    // Initialize values
    char *buffer = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: BUFFER*/
            *print_string = calloc(MAXINPUTLENGTH, 1); /** ALLOCATE: PRINT_STRING*/
    fieldNode *field = schema->fields->head;
    node *select = selects->head;
    FILE *database = fopen(schema->tableFileName, "rb");
    do
    {
        while (field != NULL)
        {
            fread(buffer, (size_t) field->length, 1, database);
            while (select != NULL)
            {
                if (strcmp(field->fieldName, select->field) == 0)
                {
                    strcpy(select->condition, buffer);
                    break;
                }
                select = select->next;
            }
            select = selects->head;
            field = field->next;
        }
        while (select != NULL)
        {
            strcat(print_string, select->condition);
            strcat(print_string, ",");
            memset(select->condition, 0, MAXINPUTLENGTH);
            select = select->next;
        }
        select = selects->head;
        trimwhitespace(print_string);
        trimChars(print_string, ",");
        if (strlen(print_string) > 0)
        {
            printf("%s\n", print_string);
        }
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(print_string, 0, MAXINPUTLENGTH);
        field = schema->fields->head;
    } while (!feof(database));
    free(buffer); /** DEALLOCATE: BUFFER*/
    free(print_string); /** DEALLOCATE: PRINT_STRING*/
}

/**
 * @brief creates a list of fields to select from a table and whether a where clause was included, then calls the
 * appropriate function call.
 * @param buffer - pointer to stdin
 */
bool selectRecord(char *buffer)
{
    char *cmd = strtok(buffer, ", ");
    cmd = strtok(NULL, ", ");
    linkedList *fields = calloc(sizeof(linkedList), 1), /** ALLOCATE: FIELDS*/
            *tables = calloc(sizeof(linkedList), 1), /** ALLOCATE: TABLES*/
            *clauses = calloc(sizeof(linkedList), 1); /** ALLOCATE: CLAUSES*/
    _table *schema = calloc(sizeof(schema), 1); /** ALLOCATE: SCHEMA*/

    // Read in comma delimited fields and create linked list of fields to select.
    while (cmd != NULL)
    {
        addNode(fields, false, cmd, "", false);
        cmd = strtok(NULL, ", ");
    }

    // Read in comma delimited tables and create linked list of tables to join.
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");
    cmd = strtok(NULL, ", ");
    while (cmd != NULL)
    {
        addNode(tables, false, cmd, "", false);
        cmd = strtok(NULL, ", ");
    }

    // Read in where clauses and create linked list of wheres to join on.
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");
    if (strcmp(cmd, "WHERE") == 0)
    {
        // Initialize fields
        char *condition, *field;
        bool constant;
        while (cmd != NULL)
        {
            // Initialize fields
            condition = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: CONDITION*/
            field = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: FIELD*/
            constant = false;

            // Create field name and string to match for where clause
            cmd = strtok(NULL, " ");
            strncat(field, cmd, MAXINPUTLENGTH);
            cmd = strtok(NULL, " =");
            if (strncmp(cmd, "\"", 1) == 0)
            {
                constant = true;
            }
            cmd = trimChars(cmd, "\"");
            strncat(condition, cmd, MAXINPUTLENGTH);
            addNode(clauses, false, field, condition, constant);
            free(field); /** DEALLOCATE: FIELD*/
            free(condition); /** DEALLOCATE: CONDITION*/
            cmd = strtok(NULL, " ");
        }
        // Read next line
        fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);

        while (tables->count > 1)
        {

        }
    }
    // Pass in fields to read without where clause info
    free(buffer);
    buffer = calloc(MAXINPUTLENGTH, 1);
    strcpy(buffer, tables->head->field);
    loadSchema(schema, buffer);
    getRecord(schema, fields);
    free(fields); /** DEALLOCATE: FIELDS*/
    free(tables); /** DEALLOCATE: TABLES*/
    free(clauses); /** DEALLOCATE: CLAUSES*/
    free(schema); /** DEALLOCATE: SCHEMA*/
}
