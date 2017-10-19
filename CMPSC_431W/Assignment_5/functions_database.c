#include "utils.h"
#include "functions_database.h"

// #############################################################################
// ### DATABASE FUNCTIONS
// #############################################################################

/**
 * @brief deletes table and schema files of database
 * @param schema_name
 * @return
 */
bool dropTable(char *schema_name)
{
    char *schema_file = calloc(MAXLENOFFIELDNAMES, sizeof(char) + 7),
            *database_file = calloc(MAXLENOFFIELDNAMES, sizeof(char) + 4);
    strcat(schema_file, schema_name);
    strcat(schema_file, ".schema");
    strcat(database_file, schema_name);
    strcat(database_file, ".bin");

    remove(schema_file);
    remove(database_file);

}

/**
 * @brief Saves data into a .schema file given a table structure for reference
 * @param table - pointer to table structure generated with loadSchema
 * @return returns true if function completes.
 */
bool loadDatabase(_table *table, char *buffer)
{
    // Initialize values
    char *record, *current,
            *filename = table->tableFileName; /** ALLOCATE: FILENAME */
    int record_length = table->reclen,
            rec_loc = 0;
    FILE *database;

    if (access(filename, F_OK) == -1)
    {
        database = fopen(filename, "wb+"); /** OPEN FILE: DATABASE */
    }
    else
    {
        database = fopen(filename, "ab"); /** OPEN FILE: DATABASE */
    }
    record = calloc(1, (size_t) record_length); /** ALLOCATE: RECORD */
    current = strtok(buffer, " ,");
    current = strtok(NULL, " ,");
    current = strtok(NULL, " ,");
    current = strtok(NULL, " ,");
    for (int i = 0; i < table->fieldcount; i++)
    {
        int f_length = table->fields[i].fieldLength;
        if (strlen(current) > f_length) // Check if field is larger than accepted value
        {
            printf("*** WARNING: Data in field %s is being truncated ***\n", table->fields[i].fieldName);
        }
        strncat(&record[rec_loc], current, (size_t) (f_length - 1));
        rec_loc += f_length; // Ensure next field is written at proper location
        current = strtok(NULL, ",");
    }
    fwrite(record, record_length - 1, 1, database);
    fwrite("\n", 1, 1, database);
    fclose(database); /** CLOSE FILE: DATABASE */
    free(filename); /** DEALLOCATE: FILENAME */
    free(record); /** DEALLOCATE: RECORD */
    return true;
}