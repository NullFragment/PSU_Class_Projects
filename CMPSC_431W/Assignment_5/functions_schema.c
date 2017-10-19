#include "utils.h"
#include "functions_schema.h"

// #############################################################################
// ### SCHEMA FUNCTIONS
// #############################################################################

/**
 * @brief loadSchema creates a table within a table struct
 * @param table - reference to table struct to use
 * @param buffer - name of schema file, excluding extension
 * @return - returns true if successful
 */

bool loadSchema(_table *table, char *buffer)
{
    // Set file name and open schema file
    char *file_name = calloc(1, MAXLENOFFIELDNAMES + 8); /** ALLOCATE: FILE NAME */
    strcat(file_name, buffer);
    strcat(file_name, ".schema");

    // Exit out if schema file does not exist
    if (access(file_name, F_OK) == -1)
    {
        // Read next line
        fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);
        file_name = strtok(file_name, ".");
        printf("Table %s does not exist.\n", file_name);
        return false;
    }

    FILE *schema = fopen(file_name, "rb"); /** OPEN FILE: SCHEMA */

    // Initialize number of fields counter and buffer string
    int field_number = 0;
    char *str_in = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: STR IN */
    fread(str_in, MAXINPUTLENGTH, 1, schema);

    // Initialize table metadata
    table->tableFileName = calloc(MAXLENOFFIELDNAMES, sizeof(char));
    strncpy(table->tableFileName, buffer, MAXLENOFFIELDNAMES);
    strcat(table->tableFileName, ".bin");
    table->reclen = 0;

    // Start reading file string and read until end of file
    do
    {
        char *current = strtok(str_in, " \n");
        if (strcmp(current, "ADD") == 0)
        {
            _field *current_field = &table->fields[field_number];
            table->fieldcount++;
            strncpy(current_field->fieldName, strtok(NULL, " \n"), MAXLENOFFIELDNAMES);
            strncpy(current_field->fieldType, strtok(NULL, " \n"), MAXLENOFFIELDTYPES);
            current_field->fieldLength = atoi(strtok(NULL, " \n"));
            table->reclen += current_field->fieldLength;
            field_number++;
        }
        free(str_in);
        str_in = calloc(MAXINPUTLENGTH, sizeof(char));
        fread(str_in, MAXINPUTLENGTH, 1, schema);
    } while (strncmp(str_in, "END", 3) != 0);
    fclose(schema); /** CLOSE FILE: SCHEMA */
    free(file_name); /** DEALLOCATE: FILE NAME */
    free(str_in); /** DEALLOCATE: STR IN */
    return true;
}

/**
 * @brief Function saves SQL add calls and saves them to .schema file.
 * @param file_name - takes name of file to be used excluding file extension
 * @param buffer - pointer to buffer for stdin
 * @return
 */
bool createSchema(char *schema_name, char *buffer, FILE *stream)
{
    // Allocate memory for and create filename
    char *file_name = calloc(1, MAXLENOFFIELDNAMES + 8); /** ALLOCATE: FILE NAME */
    strcat(file_name, schema_name);
    strcat(file_name, ".schema");


    FILE *schema = fopen(file_name, "wb+"); /** OPEN FILE: SCHEMA */
    memset(buffer, 0, MAXINPUTLENGTH);
    fgets(buffer, MAXINPUTLENGTH - 1, stream);

    // Start reading in schema structure and saving to file
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    while (strncmp(buffer, "END", 3) != 0 && buffer != NULL)
    {
        fwrite(buffer, MAXINPUTLENGTH - 1, 1, schema);
        fwrite("\n", 1, 1, schema);
        memset(buffer, 0, MAXINPUTLENGTH);
        fgets(buffer, MAXINPUTLENGTH - 1, stream);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);
    }
    fwrite("END\n", 4, 1, schema);
    fclose(schema); /** CLOSE FILE: SCHEMA */
    free(file_name); /** DEALLOCATE: FILE NAME */
}

/**
 * @brief - Parses through a given schema file and prints out records
 * @param schema - requires reference to loaded schema struct
 */
void printSchema(_table *schema)
{
    printf("----------- SCHEMA --------------\n");
    printf("TABLE NAME: %.*s\n", (int) strlen(schema->tableFileName) - 4, schema->tableFileName);
    for (int i = 0; i < schema->fieldcount; i++)
    {
        printf("--- %s (%s-%d)\n", schema->fields[i].fieldName, schema->fields[i].fieldType,
               schema->fields[i].fieldLength);
    }
}