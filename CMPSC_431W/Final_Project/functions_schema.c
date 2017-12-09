#include "functions_schema.h"
#include "functions_field_list.h"
#include "functions_linked_list.h"
#include "utils.h"
#include "functions_records.h"

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
    char *file_name = calloc(1, MAXINPUTLENGTH + 8); /** ALLOCATE: FILE NAME */
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
    if (compareStrings(str_in, "INDEX", 5, 0))
    {
        table->index = true;
        fread(str_in, MAXINPUTLENGTH, 1, schema);
    }
    else
    {
        table->index = false;
    }

    // Initialize table metadata
    table->tableFileName = calloc(MAXINPUTLENGTH, sizeof(char));
    strncpy(table->tableFileName, buffer, MAXINPUTLENGTH);
    strcat(table->tableFileName, ".bin");
    table->reclen = 0;
    table->fields = calloc(sizeof(fieldList), 1);
    // Start reading file string and read until end of file

    do
    {
        char *fieldName = calloc(MAXINPUTLENGTH, 1),
                *fieldType = calloc(MAXINPUTLENGTH, 1),
                *current = strtok(str_in, " \n");
        int fieldLength;
        if (compareStrings(current, "ADD", 3, 0))
        {
            table->fieldcount++;
            strncpy(fieldName, strtok(NULL, " \n"), MAXINPUTLENGTH);
            strncpy(fieldType, strtok(NULL, " \n"), MAXINPUTLENGTH);
            fieldLength = atoi(strtok(NULL, " \n"));
            table->reclen += fieldLength;
            addfieldNode(table->fields, false, fieldName, fieldType, fieldLength);
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
    char *file_name = calloc(1, MAXINPUTLENGTH + 8); /** ALLOCATE: FILE NAME */
    strcat(file_name, schema_name);
    strcat(file_name, ".schema");


    FILE *schema;
    if (append == true && access(file_name, F_OK) == 0)
    {
        schema = fopen(file_name, "ab+"); /** OPEN FILE: SCHEMA */
    }
    else
    {
        schema = fopen(file_name, "wb+"); /** OPEN FILE: SCHEMA */
    }
    memset(buffer, 0, MAXINPUTLENGTH);
    if (stream == stdin)
    {
        fgets(buffer, MAXINPUTLENGTH, stream);
    }
    else
    {
        fread(buffer, sizeof(char), MAXINPUTLENGTH, stream);
    }

    // Start reading in schema structure and saving to file
    trimwhitespace(buffer);
    if (logging) printf("===> %s\n", buffer);
    while (!compareStrings(buffer, "END", 3, 0) && buffer != NULL && !feof(stream))
    {
        fwrite(buffer, MAXINPUTLENGTH - 1, 1, schema);
        fwrite("\n", 1, 1, schema);
        memset(buffer, 0, MAXINPUTLENGTH);
        if (stream == stdin)
        {
            fgets(buffer, MAXINPUTLENGTH, stream);
        }
        else
        {
            fread(buffer, sizeof(char), MAXINPUTLENGTH, stream);
        }
        trimwhitespace(buffer);
        if (logging) printf("===> %s\n", buffer);
    }
    fclose(schema); /** CLOSE FILE: SCHEMA */
    free(file_name); /** DEALLOCATE: FILE NAME */
}

///**
// * @brief - Parses through a given schema file and prints out records
// * @param schema - requires reference to loaded schema struct
// */
//void printSchema(_table *schema)
//{
//    printf("----------- SCHEMA --------------\n");
//    printf("TABLE NAME: %.*s\n", (int) strlen(schema->tableFileName) - 4, schema->tableFileName);
//    for (int i = 0; i < schema->fieldcount; i++)
//    {
//        printf("--- %s (%s-%d)\n", schema->fields[i].fieldName, schema->fields[i].fieldType,
//               schema->fields[i].fieldLength);
//    }
//}

void createTempSchema(char *first, char *second, char *temp_name)
{
    FILE *table1, *table2;
    char *name_t1 = calloc(strlen(first) + 8, 1),
            *name_t2 = calloc(strlen(second) + 8, 1),
            *buffer = calloc(MAXINPUTLENGTH, 1);

    strncat(name_t1, first, strlen(first) - 4);
    strncat(name_t1, ".schema", 7);
    strncat(name_t2, second, strlen(second) - 4);
    strncat(name_t2, ".schema", 7);
    table1 = fopen(name_t1, "rb");
    table2 = fopen(name_t2, "rb");
    createSchema(temp_name, buffer, table1, false, false);
    createSchema(temp_name, buffer, table2, true, false);
}

/**
 * @brief Reads in index parameters and creates an indexed file from existing data
 * @param buffer
 * @param stream
 */
void createIndex(char *buffer, FILE *stream)
{
    char *token, *indexName, *baseTableName = NULL;
    linkedList *indexOn = calloc(sizeof(linkedList), 1); /** ALLOCATE: indexOn */
    fieldList *indexFields = calloc(sizeof(fieldList), 1); /** ALLOCATE: indexFields */
    _table *baseTable = calloc(sizeof(_table), 1); /** ALLOCATE: baseTable */
    token = strtok(buffer, " ,\n");
    token = strtok(NULL, " ,\n");
    token = strtok(NULL, " ,\n");
    indexName = calloc(strlen(token) + 8, sizeof(char)); /** ALLOCATE: indexName */
    strncpy(indexName, token, strlen(token));
    token = strtok(NULL, " ,\n");

    // Create linked list of fields to use
    if (compareStrings(token, "USING", 5, 0))
    {
        token = strtok(NULL, " ,\n");
        while (token != NULL)
        {
            addNode(indexOn, false, token, " ", 0, false);
            token = strtok(NULL, " ,\n");
        }
    }

    // Load Next Line
    fgets(buffer, MAXINPUTLENGTH - 1, stream);
    printf("===> %s", buffer);

    // Load base table into memory and create index field list
    if (compareStrings(buffer, "FROM", 4, 0))
    {
        token = strtok(buffer, " ,\n");
        token = strtok(NULL, " ,\n");
        baseTableName = calloc(strlen(token) + 1, sizeof(char)); /** ALLOCATE: baseTableName */
        strncpy(baseTableName, token, strlen(token) + 1);
        if (loadSchema(baseTable, baseTableName) == true)
        {
            fieldNode *traceBaseFields = baseTable->fields->head;
            node *traceIndexFields = indexOn->head;
            while (traceIndexFields != NULL)
            {
                while (traceBaseFields != NULL)
                {
                    if (compareStrings(traceBaseFields->fieldName, traceIndexFields->field, 0, 0))
                    {
                        addfieldNode(indexFields, false, traceBaseFields->fieldName, traceBaseFields->fieldType,
                                     traceBaseFields->length);
                    }
                    traceBaseFields = traceBaseFields->next;
                }
                traceBaseFields = baseTable->fields->head;
                traceIndexFields = traceIndexFields->next;
            }
        }
    }

    // Load Next Line
    fgets(buffer, MAXINPUTLENGTH - 1, stream);
    printf("===> %s", buffer);

    // Generate index schema file
    strncat(indexName, ".schema", 7);
    FILE *index = fopen(indexName, "wb+"); /** OPEN: index */
    fieldNode *indexField = indexFields->head;
    char *toPrint = calloc(MAXINPUTLENGTH, sizeof(char));
    strcat(toPrint, "INDEX");
    fwrite(toPrint, MAXINPUTLENGTH - 1, 1, index);
    fwrite("\n", 1, 1, index);
    while (indexField != NULL)
    {
        memset(toPrint, 0, MAXINPUTLENGTH);
        sprintf(toPrint, "ADD %s %s %d", indexField->fieldName, indexField->fieldType, indexField->length);
        fwrite(toPrint, MAXINPUTLENGTH - 1, 1, index);
        fwrite("\n", 1, 1, index);
        indexField = indexField->next;
    }
    fclose(index); /** CLOSE: index */

    if (baseTableName != NULL)
    {
        loadIndex(indexName, baseTable, indexOn, indexFields);
    }

    free(indexName); /** DEALLOCATE: indexName */
    free(indexOn); /** DEALLOCATE: indexOn */
    free(indexFields); /** DEALLOCATE: indexFields */
    free(baseTableName); /** DEALLOCATE: baseTableName */
    free(baseTable); /** DEALLOCATE: baseTable */
}

void loadIndex(char *indexName, _table *baseTable, linkedList *indexOn, fieldList *idxFields)
{
    char *indexBin = calloc(strlen(indexName) + 1, sizeof(char));
    FILE *inFile, *outFile;
    strncpy(indexBin, indexName, strlen(indexName) - 7);
    strcat(indexBin, ".bin");

    // Save all records to garbage file to read from
    inFile = fopen("garbage.bin", "w");
    getRecord(baseTable, indexOn, inFile);
    fclose(inFile);

    // Rewrite records at proper text length to sort
    inFile = fopen("garbage.bin", "r");
    outFile = fopen("garbage.txt", "w");
    parseFile(inFile, outFile, idxFields, true);
    fclose(inFile);
    fclose(outFile);

    // Sort garbage file
    system("sort garbage.txt > garbage_srt.txt");

    // re-write all records to text at proper length in bin file
    inFile = fopen("garbage_srt.txt", "r");
    outFile = fopen(indexBin, "wb+");
    parseFile(inFile, outFile, idxFields, false);
    fclose(inFile);
    fclose(outFile);
}

void parseFile(FILE *toParse, FILE *output, fieldList *fields, bool comma)
{
    char *token, *parseBuffer = calloc(MAXINPUTLENGTH, 1);
    fieldNode *trace = fields->head;

    fgets(parseBuffer, MAXINPUTLENGTH - 1, toParse);
    while (!feof(toParse))
    {
        token = strtok(parseBuffer, ",\n");
        while (trace != NULL)
        {
            trimwhitespace(token);
            if (comma == true)
            {
                fprintf(output, "%-*.*s", trace->length, trace->length, token);
                fprintf(output, ",");
            }
            else
            {
                int length = trace->length;
                if (trace->next == NULL) length--;
                fwrite(token, (size_t) length, sizeof(char), output);
            }
            token = strtok(NULL, ",\n");
            trace = trace->next;
        }
        fprintf(output, "\n");
        trace = fields->head;
        fgets(parseBuffer, MAXINPUTLENGTH - 1, toParse);
    }
}