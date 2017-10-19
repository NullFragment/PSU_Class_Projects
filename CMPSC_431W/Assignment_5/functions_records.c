#include "utils.h"
#include "functions_records.h"
#include "functions_schema.h"

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################

char *createString(_table *schema, char *buffer,)
{
    // Open schema file and search through all records for wanted information
    FILE *table = fopen(schema->tableFileName, "rb");
    strtok(buffer, " \n\0");
    while (!feof(table))
    {
        char *to_print = calloc(sizeof(char), MAXINPUTLENGTH);
        for (int i = 0; i < schema->fieldcount; i++)
        {
            fread(buffer, (unsigned) schema->fields[i].fieldLength, 1, table);
            trimwhitespace(buffer);
            if (strlen(buffer) == 0) break;
            for (int j = 0; j < field_counter; j++)
            {
                if (field_numbers[j] == i)
                {
                    trimwhitespace(buffer);
                    strcat(to_print, buffer);
                    strcat(to_print, " ");
                }
            }
            if (where_check == true && i == matching_field && strcmp(buffer, condition) == 0)
            {
                print_flag = true;
            }
            memset(buffer, 0, MAXINPUTLENGTH);
        }

        if (strlen(to_print) > 0 && (print_flag == true || where_check == false))
        {
            printf("%s\n", to_print);
            print_flag = false;
        }
        free(to_print);
    }
    fclose(table);
};


/**
 * @brief finds all records in a given populated table
 * @param schema - pointer to loaded schema
 * @param fields - pointer to comma separated string of fields
 * @param to_match - pointer to character of field to compare against for where clause
 * @param condition - pointer to value to compare with for where clause
 */
void getRecord(_table *schema, linkedList *fields, linkedList *clauses)
{
    // Initialize values
    char *buffer = calloc(MAXINPUTLENGTH, 1);
    int field_counter = 0, matching_field = -1;
    bool where_check = false, print_flag = false;
    int *field_numbers = calloc((unsigned) schema->fieldcount, sizeof(int));
    char *field = strtok(fields, ",");

    // Determine whether or not to check where clause
    if (clauses->count > 0)
    {
        where_check = true;
    }

    // Find all matching fields and create an array of their indices.
    while (field != NULL)
    {
        for (int i = 0; i < schema->fieldcount; i++)
        {
            if (strcmp(schema->fields[i].fieldName, field) == 0)
            {
                field_numbers[field_counter] = i;
                field_counter++;
            }
            if (where_check == true && matching_field < 0 && strcmp(schema->fields[i].fieldName, to_match) == 0)
            {
                matching_field = i;
            }
        }
        field = strtok(NULL, ",");
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

//        getRecord(&table, fields, field, condition);
    } else if (strcmp(cmd, "WHERE") != 0)
    {
        // Pass in fields to read without where clause info
//        getRecord(&table, fields, NULL);
    }
}
