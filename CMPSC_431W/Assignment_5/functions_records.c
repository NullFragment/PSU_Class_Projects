#include "utils.h"
#include "functions_records.h"
#include "functions_schema.h"

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################

void createTempTable(linkedList *clauses, linkedList *fields)
{
    
}

void joinTable(_table *first, _table *second, linkedList *clauses, linkedList *fields)
{
    for (int i = 0; i < first->fieldcount; i++)
    {

    }
}

/**
 * @brief creates a list of fields to select from a table and whether a where clause was included, then calls the
 * appropriate function call.
 * @param buffer - pointer to stdin
 */
bool selectRecord(char *buffer)
{
    char *cmd = strtok(NULL, ", ");
    linkedList *fields = calloc(sizeof(linkedList), 1);
    _table table;

    // Read in comma delimited fields and reconstruct search field array.
    while (cmd != NULL)
    {
        addNode(fields, false, cmd, "", false);
        cmd = strtok(NULL, ", ");
    }

    // Read next line
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");

    // Load table if it exists, if not, break early
    if (strcmp(cmd, "FROM") == 0)
    {
        cmd = strtok(NULL, " \n");
        if (!loadSchema(&table, cmd)) return false;
    }

    // Read next line
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");

    if (strcmp(cmd, "WHERE") == 0)
    {
        // Initialize fields
        char *condition, *field;
        bool constant;
        linkedList *clauses = calloc(sizeof(linkedList), 1);

        while (cmd != NULL)
        {
            // Initialize fields
            condition = calloc(MAXLENOFFIELDNAMES, sizeof(char));
            field = calloc(MAXINPUTLENGTH, sizeof(char));
            constant = false;

            // Create field name and string to match for where clause
            cmd = strtok(NULL, " ");
            strncat(field, cmd, MAXLENOFFIELDNAMES);
            cmd = strtok(NULL, " =");
            if (strncmp(cmd, "\"", 1) == 0)
            {
                constant = true;
            }
            cmd = trimQuotes(cmd);
            strncat(condition, cmd, MAXINPUTLENGTH);
            addNode(clauses, false, field, condition, constant);
            free(field);
            free(condition);
            cmd = strtok(NULL, " ");
        }


        // Read next line
        fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);

        //getRecord(&table, fields, field, condition);
    }
    else if (strcmp(cmd, "WHERE") != 0)
    {
        // Pass in fields to read without where clause info
        getRecord(&table, fields, NULL);
    }
}
