#include "utils.h"
#include "functions_records.h"
#include "functions_schema.h"
#include "functions_linked_list.h"
#include "functions_database.h"

// #############################################################################
// ### INDEX SEARCH FUNCTIONS
// #############################################################################

void getIndexedRecord(_table *schema, linkedList *selects, linkedList *clauses, FILE *output)
{
    FILE *database;
    char *buffer = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: BUFFER */
            *print_string = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: PRINT_STRING */
            *db_name = calloc(strlen(schema->tableFileName) + 5, 1);
    int compareVal = 0, iter = 0;
    fieldNode *field = schema->fields->head;
    node *select = selects->head,
            *clause = clauses->head;
    strcat(db_name, "temp_");
    strcat(db_name, schema->tableFileName);
    if (access(db_name, F_OK) != -1)
    {
        database = fopen(db_name, "rb"); /** OPEN: DATABASE */
    } else
    {
        database = fopen(schema->tableFileName, "rb"); /** OPEN: DATABASE */
    }

    // Get number of records in file
    fseek(database, 0L, SEEK_END);
    long fileLen = ftell(database);
    long records = fileLen / (schema->reclen);
    long seekLen = (records / 2);
    rewind(database);
    do
    {
        compareVal = 0;
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(print_string, 0, MAXINPUTLENGTH);
        fseek(database, seekLen * schema->reclen, SEEK_CUR);
        while (field != NULL)
        {
            fread(buffer, (size_t) field->length, 1, database);
            strncat(print_string, buffer, (size_t) field->length);
            strncat(print_string, ",", 1);
            field = field->next;
        }
        field = schema->fields->head;
        trimChars(print_string, ",");
        fprintf(output, "==> TRACE: %s\n", print_string);
        compareVal = strncmp(print_string, clause->condition, strlen(clause->condition));
        fseek(database, -1 * schema->reclen, SEEK_CUR);
        if (compareVal < 0)
        {
            seekLen = seekLen / 2;
            if(seekLen == 0) seekLen = 1;
        } else if (compareVal > 0)
        {
            seekLen = -1 * seekLen / 2;
            if(seekLen == 0) seekLen = -1;
        }
        if (compareVal == 0 && strlen(print_string) > 0)
        {
            fprintf(output, "%s\n", print_string);
        }
    } while (compareVal != 0 && iter < MAXBINSEARCH);

}

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################

/**
 * @brief getRecord finds all of the fields requested and prints them out
 * @param schema - reference to loaded table
 * @param selects - reference to linked list of fields to get
 */
void getRecord(_table *schema, linkedList *selects, FILE *output)
{
    // Initialize values
    FILE *database;
    char *buffer = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: BUFFER */
            *print_string = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: PRINT_STRING */
            *db_name = calloc(strlen(schema->tableFileName) + 5, 1);
    fieldNode *field = schema->fields->head;
    node *select = selects->head;
    strcat(db_name, "temp_");
    strcat(db_name, schema->tableFileName);
    if (access(db_name, F_OK) != -1)
    {
        database = fopen(db_name, "rb"); /** OPEN: DATABASE */
    } else
    {
        database = fopen(schema->tableFileName, "rb"); /** OPEN: DATABASE */
    }
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
            fprintf(output, "%s\n", print_string);
        }
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(print_string, 0, MAXINPUTLENGTH);
        field = schema->fields->head;
    } while (!feof(database));
    fclose(database);  /** CLOSE: DATABASE */
    free(buffer); /** DEALLOCATE: BUFFER */
    free(print_string); /** DEALLOCATE: PRINT_STRING */
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
    linkedList *fields = calloc(sizeof(linkedList), 1), /** ALLOCATE: FIELDS */
            *tables = calloc(sizeof(linkedList), 1), /** ALLOCATE: TABLES */
            *clauses = calloc(sizeof(linkedList), 1); /** ALLOCATE: CLAUSES */
    _table *schema = calloc(sizeof(schema), 1); /** ALLOCATE: SCHEMA */

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
        while (strncmp(cmd, "END", 3) != 0)
        {
            // Initialize fields
            condition = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: CONDITION */
            field = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: FIELD */
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
            free(field); /** DEALLOCATE: FIELD */
            free(condition); /** DEALLOCATE: CONDITION */
            fgets(buffer, MAXINPUTLENGTH - 1, stdin);
            trimwhitespace(buffer);
            printf("===> %s\n", buffer);
            cmd = strtok(buffer, " ");
        }
        // Read next line
        memset(buffer, 0, MAXINPUTLENGTH);
        if (tables->count > 1 && clauses->count > 0)
        {
            int temp_count = 0;
            while (tables->count > 1)
            {
                temp_count++;
                char *temp_name = calloc(4, 1);
                sprintf(temp_name, "tt_%d", temp_count);
                _table *join1 = calloc(sizeof(_table), 1), /** ALLOCATE: JOIN1 */
                        *join2 = calloc(sizeof(_table), 1); /** ALLOCATE: JOIN2 */

                // Load both schemas in to join
                strcpy(buffer, tables->head->field);
                loadSchema(join1, buffer);
                memset(buffer, 0, MAXINPUTLENGTH);
                strcpy(buffer, tables->head->next->field);
                loadSchema(join2, buffer);

                // Eliminate unnecessary records from each database first
                checkWhereLiteral(join1, tables->head, clauses);
                checkWhereLiteral(join2, tables->head->next, clauses);

                // Create temp table with join
                joinTable(join1, join2, clauses, temp_name);

                // Update Linked List
                popNode(tables);
                popNode(tables);
                addNode(tables, true, temp_name, "", false);

                free(join1); /** DEALLOCATE: JOIN1 */
                free(join2); /** DEALLOCATE: JOIN2 */
            }
        }
        memset(buffer, 0, MAXINPUTLENGTH);
        strcpy(buffer, tables->head->field);
        loadSchema(schema, buffer);
        if (tables->count == 1 && clauses->count > 0 && schema->index == false)
        {
            checkWhereLiteral(schema, tables->head, clauses);
        } else if (tables->count == 1 && clauses->count > 0 && schema->index == true)
        {
            getIndexedRecord(schema, fields, clauses, stdout);
        }

    }
    // Pass in fields to read without where clause info
    memset(buffer, 0, MAXINPUTLENGTH);
    strcpy(buffer, tables->head->field);
    loadSchema(schema, buffer);
    if (clauses->count == 0)
    {
        getRecord(schema, fields, stdout);
    }
    free(fields); /** DEALLOCATE: FIELDS */
    free(tables); /** DEALLOCATE: TABLES */
    free(clauses); /** DEALLOCATE: CLAUSES */
}
