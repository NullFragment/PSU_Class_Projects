#include "utils.h"
#include "functions_database.h"
#include "functions_schema.h"

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
    char *schema_file = calloc(MAXINPUTLENGTH, sizeof(char) + 7),
            *database_file = calloc(MAXINPUTLENGTH, sizeof(char) + 4);
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
    fieldNode *field = table->fields->head;
    while(field != NULL)
    {
        int f_length = field->length;
        if (strlen(current) > f_length) // Check if field is larger than accepted value
        {
            printf("*** WARNING: Data in field %s is being truncated ***\n", field->fieldName);
        }
        strncat(&record[rec_loc], current, (size_t) (f_length - 1));
        rec_loc += f_length; // Ensure next field is written at proper location
        current = strtok(NULL, ",");
        field = field->next;
    }
    fwrite(record, (size_t) record_length - 1, 1, database);
    fwrite("\n", 1, 1, database);
    fclose(database); /** CLOSE FILE: DATABASE */
    free(filename); /** DEALLOCATE: FILENAME */
    free(record); /** DEALLOCATE: RECORD */
    return true;
}

/**
 * @brief Parses through database and creates a temp .bin file with all records matching a where clause on a string
 * @param schema - reference to loaded schema to check
 * @param table - reference to node of a linked list with table name
 * @param clauses - reference to linked list of where clauses
 * @return false if schema does not exist, true if function exits
 */
bool checkWhereLiteral(_table *schema, node *table, linkedList *clauses)
{
    char *buffer = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: BUFFER */
            *data_string, *temp_db_name;
    FILE *database, *temp_db;
    strcpy(buffer, table->field);
    if (loadSchema(schema, buffer) == false)
    {
        free(buffer); /** DEALLOCATE: BUFFER */
        return false;
    }

    // Initialize Variables
    fieldNode *field = schema->fields->head;
    node *where = clauses->head;
    bool failure = false;
    data_string = calloc(MAXINPUTLENGTH, 1); /** ALLOCATE: DATA_STRING */
    temp_db_name = calloc(strlen(schema->tableFileName) + 5, 1); /** ALLOCATE: TEMP_DB_NAME */
    database = fopen(schema->tableFileName, "rb"); /** OPEN: DATABASE */
    strcat(temp_db_name, "temp_");
    strcat(temp_db_name, schema->tableFileName);
    temp_db = fopen(temp_db_name, "wb+"); /** OPEN: TEMP_DB */

    // Check where clauses
    do
    {
        int stringLoc = 0;
        while (field != NULL)
        {
            fread(buffer, (size_t) field->length, 1, database);
            while (where != NULL && failure == false)
            {
                if (strcmp(field->fieldName, where->field) == 0 && strcmp(buffer, where->condition) != 0)
                {
                    failure = true;
                    break;
                }
                where = where->next;
            }
            if (failure == false)
            {
                strncat(&data_string[stringLoc], buffer, (size_t) field->length);
                stringLoc += field->length;
            }
            where = clauses->head;
            field = field->next;
        }
        if (failure == false)
        {
            trimwhitespace(data_string);
            trimChars(data_string, ",");
            if (strlen(data_string) > 0)
            {
                fwrite(data_string, (size_t) schema->reclen - 1, 1, temp_db);
                fwrite("\n", 1, 1, temp_db);
            }
        }
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(data_string, 0, MAXINPUTLENGTH);
        field = schema->fields->head;
        failure = false;
    } while (!feof(database));
    memset(temp_db_name, 0, strlen(temp_db_name));
    fclose(database); /** CLOSE: DATABASE */
    fclose(temp_db); /** CLOSE: DATABASE */
    free(buffer); /** DEALLOCATE: BUFFER */
    free(data_string); /** DEALLOCATE: DATA_STRING */
    return true;
}

/**
 * @brief joinTable creates a temporary schema and bin file based on join clauses of two tables
 * @param first - reference to first loaded schema to join on
 * @param second - reference to second loaded schema to join on
 * @param clauses - reference to linked list of where clauses
 * @param temp_name - refernce to string containing name of temporary table
 * @return - true if successful
 */
bool joinTable(_table *first, _table *second, linkedList *clauses, char *temp_name)
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