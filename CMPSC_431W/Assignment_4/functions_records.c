#include "utils.h"
#include "functions_records.h"
#include "functions_schema.h"

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################


void getRecord(struct _table *schema, char *fields, char *to_match, char *condition)
{
    // Initialize values
    char *buffer = calloc(MAXINPUTLENGTH, 1);
    int field_counter = 0, matching_field = -1;
    bool where_check = false, print_flag = false;
    int *field_numbers = calloc((unsigned) schema->fieldcount, sizeof(int));
    char *field = strtok(fields, ",");

    // Determine whether or not to check where clause
    if (strlen(to_match) > 0)
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
                break;
            }
            if (where_check == true && matching_field < 0 && strcmp(schema->fields[i].fieldName, to_match) == 0)
            {
                matching_field = i;
            }
        }
        field = strtok(NULL, ",");
    }

    // Open schema file and search through all records for wanted information
    FILE *table = fopen(schema->tableFileName, "rb");
    strtok(buffer, " \n\0");
    while (!feof(table))
    {
        char *to_print = calloc(sizeof(char), MAXINPUTLENGTH);
        for (int i = 0; i < schema->fieldcount; i++)
        {
            fread(buffer, (unsigned) schema->fields[i].fieldLength, 1, table);
            for (int j = 0; j < field_counter; j++)
            {
                if (field_numbers[j] == i)
                {
                    strcat(to_print, buffer);
                    strcat(to_print, " ");
                }
                if (where_check == true && field_numbers[j] == matching_field && strcmp(buffer, condition) == 0)
                {
                    print_flag = true;
                }
            }
            memset(buffer, 0, MAXINPUTLENGTH);
        }

        if (print_flag == true || where_check == false)
        {
            printf("%s\n", to_print);
            print_flag = false;
        }
        free(to_print);
    }
    fclose(table);
}


/**
 * @brief creates a list of fields to select from a table and whether a where clause was included, then calls the
 * appropriate function call.
 * @param buffer - pointer to stdin
 */
void selectRecord(char *buffer)
{
    char *cmd = strtok(NULL, ", ");
    char *fields = calloc(MAXINPUTLENGTH, 1);
    struct _table table;
    while (cmd != NULL)
    {
        strncat(fields, cmd, MAXINPUTLENGTH - strlen(fields) - 1);
        strcat(fields, ",");
        cmd = strtok(NULL, ", ");
    }
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");
    if (strcmp(cmd, "FROM") == 0)
    {
        cmd = strtok(NULL, " \n");
        loadSchema(&table, cmd);
    }
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");
    if (strcmp(cmd, "WHERE") == 0)
    {
        fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);
        char *condition = calloc(MAXLENOFFIELDNAMES, sizeof(char))
        , *field = calloc(MAXINPUTLENGTH, sizeof(char));
        cmd = strtok(NULL, " ");
        strncat(field, cmd, MAXLENOFFIELDNAMES);
        cmd = strtok(NULL, " =");
        cmd = trimQuotes(cmd);
        strncat(condition, cmd, MAXLENOFFIELDNAMES);
        getRecord(&table, fields, field, condition);
    } else if (strcmp(cmd, "WHERE") != 0)
    {
        fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);
        getRecord(&table, fields, "", "");
    }
}
