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
    while (field != NULL)
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
                size_t fieldLen = strlen(field->fieldName);
                size_t whereLen = strlen(where->field);
                size_t compLen = (fieldLen > whereLen) ? fieldLen : whereLen;
                if (where->constant == true &&
                    compareStrings(field->fieldName, where->field, compLen, 0) &&
                    !compareStrings(buffer, where->compareVal, compLen, where->conditional))
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
    char *firstBuffer = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: FIRST BUFFER */
            *secondBuffer = calloc(MAXINPUTLENGTH, 1); /** ALLOCATE: SECOND BUFFER */
    _table *tempSchema = calloc(sizeof(_table), 1); /** ALLOCATE: TEMP */
    fieldNode *firstField = first->fields->head,
            *secondField = second->fields->head;
    node *where = clauses->head;
    FILE *firstDB, *secondDB, *tempDB;
    bool failure = false;

    // Create temporary names to check for temp tables
    char *firstTempDB = calloc(1, strlen(first->tableFileName) + 5);
    strcat(firstTempDB, "temp_");
    strcat(firstTempDB, first->tableFileName);
    char *secondTempDB = calloc(1, strlen(second->tableFileName) + 5);
    strcat(secondTempDB, "temp_");
    strcat(secondTempDB, second->tableFileName);


    createTempSchema(first->tableFileName, second->tableFileName, temp_name);
    strcat(firstBuffer, temp_name);
    loadSchema(tempSchema, firstBuffer);
    memset(firstBuffer, 0, MAXINPUTLENGTH);
    strcat(firstBuffer, temp_name);
    strcat(firstBuffer, ".bin");
    // Open Files
    if (access(firstTempDB, F_OK) != -1) firstDB = fopen(firstTempDB, "rb"); /** OPEN: FIRST DB */
    else firstDB = fopen(first->tableFileName, "rb");
    if (access(firstTempDB, F_OK) != -1) secondDB = fopen(secondTempDB, "rb"); /** OPEN: SECOND DB */
    else secondDB = fopen(second->tableFileName, "rb");
    tempDB = fopen(firstBuffer, "wb+");  /** OPEN: TEMP DB */


    memset(firstBuffer, 0, MAXINPUTLENGTH);

    size_t readFirst = fread(firstBuffer, 1, (size_t) firstField->length, firstDB),
            readSecond = fread(secondBuffer, 1, (size_t) secondField->length, secondDB);
    while (readFirst > 0)
    {
        while (readSecond > 0)
        {
            while (firstField != NULL)
            {
                while (secondField != NULL)
                {
                    while (where != NULL && failure == false)
                    {
                        if (where->constant == false)
                        {
                            size_t firstFieldLen = strlen(firstField->fieldName);
                            size_t secondFieldLen = strlen(secondField->fieldName);
                            size_t whereLen = strlen(where->field);
                            size_t compareLen = strlen(where->compareVal);
                            size_t firstBufLen = strlen(firstBuffer);
                            size_t secondBufLen = strlen(secondBuffer);


                            size_t firstCompLen = (firstFieldLen > whereLen) ? firstFieldLen : whereLen;
                            size_t secCompLen = (secondFieldLen > whereLen) ? secondFieldLen : whereLen;

                            size_t firstValLen = (firstFieldLen > compareLen) ? firstFieldLen : compareLen;
                            size_t secValLen = (secondFieldLen > compareLen) ? secondFieldLen : compareLen;

                            size_t bufferCompLen = (firstBufLen > secondBufLen) ? firstBufLen : secondBufLen;

                            if (compareStrings(where->field, firstField->fieldName, firstCompLen, 0) ||
                                compareStrings(where->field, secondField->fieldName, secCompLen, 0))
                            {
                                if (compareStrings(where->compareVal, firstField->fieldName, firstValLen, 0) ||
                                    compareStrings(where->compareVal, secondField->fieldName, secValLen, 0))
                                {
                                    if (!compareStrings(firstBuffer, secondBuffer, bufferCompLen, where->conditional))
                                    {
                                        failure = true;
                                    }
                                }
                            }
                        }
                        where = where->next;
                    }
                    where = clauses->head;
                    secondField = secondField->next;
                    if (secondField != NULL)
                    {
                        fread(secondBuffer, 1, (size_t) secondField->length, secondDB);
                    }
                } // End of record in DB2
                secondField = second->fields->head;
                firstField = firstField->next;
                fseek(secondDB, -second->reclen, SEEK_CUR);
                if (firstField != NULL)
                {
                    fread(firstBuffer, 1, (size_t) firstField->length, firstDB);
                    fread(secondBuffer, 1, (size_t) secondField->length, secondDB);
                }
            } // End of record in DB1

            if (failure == false)
            {
                fseek(firstDB, -first->reclen, SEEK_CUR);
                fread(firstBuffer, 1, (size_t) first->reclen, firstDB);
                fread(secondBuffer, 1, (size_t) second->reclen, secondDB);
                fwrite(firstBuffer, 1, (size_t) first->reclen - 1, tempDB);
                fwrite("\0", 1, 1, tempDB);
                fwrite(secondBuffer, 1, (size_t) second->reclen, tempDB);
            }
            else
            {
                fseek(secondDB, second->reclen, SEEK_CUR);
            }
            fseek(firstDB, -first->reclen, SEEK_CUR);
            failure = false;
            firstField = first->fields->head;
            memset(firstBuffer, 0, MAXINPUTLENGTH);
            memset(secondBuffer, 0, MAXINPUTLENGTH);
            fread(firstBuffer, 1, (size_t) firstField->length, firstDB);
            readSecond = fread(secondBuffer, 1, (size_t) secondField->length, secondDB);
        } // End of Second DB file
        fseek(firstDB, first->reclen - firstField->length, SEEK_CUR);
        rewind(secondDB);
        readSecond = fread(secondBuffer, 1, (size_t) secondField->length, secondDB);
        readFirst = fread(firstBuffer, 1, (size_t) firstField->length, firstDB);
    } // End of First DB file
    fclose(firstDB);  /** CLOSE: FIRST DB */
    fclose(secondDB);  /** CLOSE: SECOND DB */
    fclose(tempDB);  /** CLOSE: TEMP DB */
    free(firstBuffer); /** DEALLOCATE: FIRST BUFFER */
    free(secondBuffer); /** DEALLOCATE: FIRST BUFFER */
    free(tempSchema); /** DEALLOCATE: TEMP */

}
