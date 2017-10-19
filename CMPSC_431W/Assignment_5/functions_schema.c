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
        if (strncmp(current, "ADD", 3) == 0)
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
    } while (!feof(schema));
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
bool createSchema(char *schema_name, char *buffer, FILE *stream, bool append, bool logging)
{
    // Allocate memory for and create filename
    char *file_name = calloc(1, MAXLENOFFIELDNAMES + 8); /** ALLOCATE: FILE NAME */
    strcat(file_name, schema_name);
    strcat(file_name, ".schema");


    FILE *schema;
    if (append == true && access(file_name, F_OK) == 0)
    {
        schema = fopen(file_name, "ab+"); /** OPEN FILE: SCHEMA */
    } else
    {
        schema = fopen(file_name, "wb+"); /** OPEN FILE: SCHEMA */
    }
    memset(buffer, 0, MAXINPUTLENGTH);
    if (stream == stdin)
    {
        fgets(buffer, MAXINPUTLENGTH, stream);
    } else
    {
        fread(buffer, sizeof(char), MAXINPUTLENGTH, stream);
    }

    // Start reading in schema structure and saving to file
    trimwhitespace(buffer);
    if(logging) printf("===> %s\n", buffer);
    while (strncmp(buffer, "END", 3) != 0 && buffer != NULL && !feof(stream))
    {
        fwrite(buffer, MAXINPUTLENGTH - 1, 1, schema);
        fwrite("\n", 1, 1, schema);
        memset(buffer, 0, MAXINPUTLENGTH);
        if (stream == stdin)
        {
            fgets(buffer, MAXINPUTLENGTH, stream);
        } else
        {
            fread(buffer, sizeof(char), MAXINPUTLENGTH, stream);
        }
        trimwhitespace(buffer);
        if(logging) printf("===> %s\n", buffer);
    }
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

void createTempSchema(char *first, char *second, char *temp_name)
{
    FILE *table1, *table2;
    char *name_t1 = calloc(strlen(first) + 8, 1),
            *name_t2 = calloc(strlen(second) + 8, 1),
            *buffer = calloc(MAXINPUTLENGTH, 1);

    strncat(name_t1, first, strlen(first));
    strncat(name_t1, ".schema", 7);
    strncat(name_t2, second, strlen(second));
    strncat(name_t2, ".schema", 7);
    table1 = fopen(name_t1, "rb");
    table2 = fopen(name_t2, "rb");
    createSchema(temp_name, buffer, table1, false, false);
    createSchema(temp_name, buffer, table2, true, false);
}

void joinTable(_table *first, _table *second, linkedList *clauses, char *temp_name)
{
    createTempSchema(first->tableFileName, second->tableFileName, temp_name);
    _table *temp = calloc(sizeof(_table), 1);
    loadSchema(temp, temp_name);
    for (int i = 0; i < first->fieldcount; i++)
    {
        for (int j = 0; j < second->fieldcount; j++)
        {
            for (int k = 0; k < clauses->count; k++)
            {

            }
        }

    }
}