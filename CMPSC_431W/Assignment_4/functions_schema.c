#include "utils.h"
#include "functions_schema.h"

// #############################################################################
// ### SCHEMA FUNCTIONS
// #############################################################################

/**
 * @brief loadSchema creates a table within a table struct
 * @param table - reference to table struct to use
 * @param schema_name - name of schema file, excluding extension
 * @return - returns true if successful
 */

bool loadSchema(struct _table *table, char *schema_name)
{
    // Set file name and open schema file
    char *filename = calloc(1, strlen(schema_name));
    memcpy(filename, schema_name, strlen(schema_name));
    strcat(filename, ".schema");
    if (access(filename, F_OK) == -1) return false;
    FILE *schema = fopen(filename, "rb");

    // Initialize number of fields counter and buffer string
    int field_number = 0;
    char *str_in = calloc(MAXINPUTLENGTH, sizeof(char));
    fread(str_in, MAXINPUTLENGTH - 1, 1, schema);

    // Print log statements and initialize table metadata
    printf("*** LOG: Loading table fields...\n");
    table->tableFileName = calloc(MAXLENOFFIELDNAMES, sizeof(char));
    strncpy(table->tableFileName, schema_name, MAXLENOFFIELDNAMES);
    strcat(table->tableFileName, ".bin");
    printf("*** LOG: Table data name is [%s]\n", table->tableFileName);
    table->reclen = 0;

    // Start reading file string and read until end of file
    do
    {
        char *current = strtok(str_in, " \n");
        if (strcmp(current, "ADD") == 0)
        {
            struct _field *current_field = &table->fields[field_number];
            table->fieldcount++;
            strncpy(current_field->fieldName, strtok(NULL, " \n"), MAXLENOFFIELDNAMES);
            strncpy(current_field->fieldType, strtok(NULL, " \n"), MAXLENOFFIELDTYPES);
            current_field->fieldLength = atoi(strtok(NULL, " \n"));
            table->reclen += current_field->fieldLength;
            //printf("*** LOG: ADDING FIELD [%s] [%s] [%d]\n",
            //       current_field->fieldName, current_field->fieldType, current_field->fieldLength);
            field_number++;
        }
        memset(str_in, 0, MAXINPUTLENGTH);
        fread(str_in, MAXINPUTLENGTH - 1, 1, schema);
    } while (strlen(str_in) > 3);
    fclose(schema);
    free(str_in);
    printf("*** LOG: Table schema name is [%s]\n", filename);
    printf("*** LOG: END OF CREATE TABLE\n");
    return true;
}
/**
 * @brief Function saves SQL add calls and saves them to .schema file.
 * @param file_name - takes name of file to be used excluding file extension
 * @param buffer - pointer to buffer for stdin
 * @return
 */
bool createSchema(char *file_name, char *buffer)
{
char *schema_name = calloc(1, strlen(file_name + 1));
memcpy(schema_name, file_name, strlen(file_name));
strcat(file_name, ".schema");
/*
// UNCOMMENT TO NOT OVERWRITE SCHEMA FILES
if(access(filename, F_OK) == -1)
{
*/
printf("*** LOG: Creating table...\n");
FILE *schema = fopen(file_name, "wb+");
memset(buffer, 0, MAXINPUTLENGTH);
fgets(buffer, MAXINPUTLENGTH - 1, stdin);
while (strncmp(buffer, "END", 3) != 0 && buffer != NULL)
{
fwrite(buffer, MAXINPUTLENGTH - 1, 1, schema);
memset(buffer, 0, MAXINPUTLENGTH);
fgets(buffer, MAXINPUTLENGTH - 1, stdin);
}
fclose(schema);
struct _table table;
/*
// UNCOMMENT TO NOT OVERWRITE SCHEMA FILES
}
*/
}

/**
 * @brief - Parses through a given schema file and prints out records
 * @param schema - requires reference to loaded schema struct
 */
void printSchema(struct _table *schema)
{
    printf("----------- SCHEMA --------------\n");
    printf("TABLE NAME: %.*s\n", (int) strlen(schema->tableFileName) - 4, schema->tableFileName);
    for (int i = 0; i < schema->fieldcount; i++)
    {
        printf("--- %s (%s-%d)\n", schema->fields[i].fieldName, schema->fields[i].fieldType,
               schema->fields[i].fieldLength);
    }
}