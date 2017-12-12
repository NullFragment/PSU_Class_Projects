#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#define MAXINPUTLENGTH 5000
#define MAXBINSEARCH 100

typedef enum
{
    false, true
} bool;

typedef struct node
{
    char *field;
    char *compareVal;
    int conditional;
    bool constant;
    struct node *next;
} node;

typedef struct
{
    node *head;
    node *tail;
    int count;
} linkedList;

typedef struct fieldNode
{
    char *fieldName;
    char *fieldType;
    int length;
    struct fieldNode *next;
} fieldNode;

typedef struct
{
    fieldNode *head;
    fieldNode *tail;
    int count;
} fieldList;

typedef struct
{
    char *tableFileName;
    int reclen;
    int fieldcount;
    bool index;
    fieldList *fields;
} _table;

void trimwhitespace(char *to_trim);

char *trimChars(char *string, char *to_trim);

bool compareStrings(char *string1, char *string2, size_t length, int comparison);

bool loadSchema(_table *table, char *buffer);

bool createSchema(char *schema_name, char *buffer, FILE *stream, bool append, bool logging);

void createTempSchema(char *first, char *second, char *temp_name);

void createIndex(char *buffer, FILE *stream);

void loadIndex(char *indexName, _table *baseTable, linkedList *indexOn, fieldList *idxFields);

void parseFile(FILE *toParse, FILE *output, fieldList *fields, bool comma);

void getIndexedRecord(_table *schema, linkedList *selects, linkedList *clauses, FILE *output);

void getRecord(_table *schema, linkedList *selects, FILE *output);

bool selectRecord(char *buffer);

bool dropTable(char *schema_name);

bool loadDatabase(_table *table, char *buffer);

bool checkWhereLiteral(_table *schema, node *table, linkedList *clauses);

bool joinTable(_table *first, _table *second, linkedList *clauses, char *temp_name);


// #############################################################################
// ### UTILITY FUNCTIONS
// #############################################################################

/**
 * @brief Trims whitespace from a given character array
 * @param to_trim - pointer to array to trim whitespace from
 */
void trimwhitespace(char *to_trim)
{
    char *j;
    while (isspace(*to_trim))
    {
        to_trim++;
    }
    size_t length = strlen(to_trim);
    j = to_trim + length - 1;
    while (isspace(*j))
    {
        *j = 0;
        j--;
    }
}

/**
 * @brief Trims quotes from a given character array
 * @param string - pointer to array to trim whitespace from
 */
char *trimChars(char *string, char *to_trim)
{
    char *j;
    while (strncmp(string, to_trim, 1) == 0)
    {
        string++;
    }
    size_t length = strlen(string);
    j = string + length - 1;
    while (strcmp(j, to_trim) == 0)
    {
        *j = 0;
        j--;
    }
    return string;
}

bool compareStrings(char *string1, char *string2, size_t length, int comparison)
{
    if (length != 0)
    {
        switch (comparison)
        {
            case 0:
            {
                if (strncmp(string1, string2, length) == 0) return true;
                else return false;
            }
            case -1:
            {
                if (strncmp(string1, string2, length) < 0) return true;
                else return false;
            }
            case -2:
            {
                if (strncmp(string1, string2, length) <= 0)
                    return true;
                else return false;
            }
            case 1:
            {
                if (strncmp(string1, string2, length) > 0) return true;
                else return false;
            }
            case 2:
            {
                if (strncmp(string1, string2, length) >= 0) return true;
                else return false;
            }
            default:
            {
                fprintf(stderr, "Invalid case entered -- returning failure");
                return false;
            }
        }
    }
    else
    {
        switch (comparison)
        {
            case 0:
            {
//                int cmpVal = strcmp(string1, string2);
//                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) == 0) return true;
                else return false;
            }
            case -1:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) < 0) return true;
                else return false;
            }
            case -2:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) <= 0) return true;
                else return false;
            }
            case 1:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) > 0) return true;
                else return false;
            }
            case 2:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) >= 0) return true;
                else return false;
            }
            default:
            {
                fprintf(stderr, "Invalid case entered -- returning failure");
                return false;
            }
        }
    }
}

// #############################################################################
// ### LINKED LIST FUNCTIONS
// #############################################################################

void fillNode(node *to_fill, char *field, char *compareVal, int conditional, bool constant)
{
    to_fill->compareVal = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->compareVal, compareVal, MAXINPUTLENGTH - 1);
    to_fill->field = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->field, field, MAXINPUTLENGTH - 1);
    to_fill->constant = constant;
    to_fill->conditional = conditional;
}


linkedList *makeLinkedList(char *field, char *compareVal, int conditional, bool constant)
{
    linkedList *list = (linkedList *) calloc(sizeof(linkedList), 1);
    list->count = 1;
    list->head = calloc(sizeof(node), 1);
    list->tail = list->head;
    fillNode(list->head, field, compareVal, conditional, constant);
    return list;
}

bool addNode(linkedList *list, bool at_head, char *field, char *compareVal, int condition, bool constant)
{
    if (at_head == false && list->tail != NULL)
    {
        list->tail->next = calloc(sizeof(node), 1);
        list->tail = list->tail->next;
        fillNode(list->tail, field, compareVal, condition, constant);
        list->count++;
        return true;
    }
    else if (at_head == true && list->head != NULL)
    {
        node *temp = calloc(sizeof(node), 1);
        temp->next = list->head;
        list->head = temp;
        fillNode(list->head, field, compareVal, condition, constant);
        list->count++;
        return true;
    }
    else if (list->count == 0 && list->head == NULL)
    {
        list->head = calloc(sizeof(node), 1);
        fillNode(list->head, field, compareVal, condition, constant);
        list->count++;
        if (list->tail == NULL)
        {
            list->tail = list->head;
        }
        return true;
    }
    else return false;
}

void popNode(linkedList *list)
{
    if (list->count > 0 && list->head != NULL)
    {
        node *temp = list->head;
        list->head = list->head->next;
        free(temp);
        list->count--;
    }
    if (list->count == 0)
    {
        list->tail = NULL;
    }
}

// #############################################################################
// ### FIELD LIST FUNCTIONS
// #############################################################################

void fillFieldNode(fieldNode *to_fill, char *fieldName, char *fieldType, int length)
{
    to_fill->length = length;
    to_fill->fieldName = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->fieldName, fieldName, MAXINPUTLENGTH - 1);
    to_fill->fieldType = calloc(MAXINPUTLENGTH, 1);
    strncpy(to_fill->fieldType, fieldType, MAXINPUTLENGTH - 1);
}


fieldList *makeFieldList(char *fieldName, char *fieldType, int length)
{
    fieldList *list = (fieldList *) calloc(sizeof(fieldList), 1);
    list->count = 1;
    list->head = calloc(sizeof(fieldNode), 1);
    list->tail = list->head;
    fillFieldNode(list->head, fieldName, fieldType, length);
    return list;
}

bool addfieldNode(fieldList *list, bool at_head, char *fieldName, char *fieldType, int length)
{
    if (at_head == false && list->tail != NULL)
    {
        list->tail->next = calloc(sizeof(fieldNode), 1);
        list->tail = list->tail->next;
        fillFieldNode(list->tail, fieldName, fieldType, length);
        list->count++;
        return true;
    } else if (at_head == true && list->head != NULL)
    {
        fieldNode *temp = calloc(sizeof(fieldNode), 1);
        temp->next = list->head;
        list->head = temp;
        fillFieldNode(list->head, fieldName, fieldType, length);
        list->count++;
        return true;
    } else if (list->count == 0 && list->head == NULL)
    {
        list->head = calloc(sizeof(fieldNode), 1);
        fillFieldNode(list->head, fieldName, fieldType, length);
        list->count++;
        if (list->tail == NULL)
        {
            list->tail = list->head;
        }
        return true;
    } else return false;
}

void popFieldNode(fieldList *list)
{
    if (list->count > 0 && list->head != NULL)
    {
        fieldNode *temp = list->head;
        list->head = list->head->next;
        free(temp);
        list->count--;
    }
    if (list->count == 0)
    {
        list->tail = NULL;
    }
}

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

// #############################################################################
// ### INDEX SEARCH FUNCTIONS
// #############################################################################

void getIndexedRecord(_table *schema, linkedList *selects, linkedList *clauses, FILE *output)
{
    FILE *database;
    char *buffer = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: BUFFER */
            *print_string = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: PRINT_STRING */
            *db_name = calloc(strlen(schema->tableFileName) + 5, 1);
    int compareVal = 0, iter = 0;
    fieldNode *field = schema->fields->head;
    node *select = selects->head,
            *clause = clauses->head;
    strcat(db_name, "temp_");
    strcat(db_name, schema->tableFileName);
    if (access(db_name, F_OK) != -1)
    {
        database = fopen(db_name, "rb"); /** OPEN: DATABASE */
    }
    else
    {
        database = fopen(schema->tableFileName, "rb"); /** OPEN: DATABASE */
    }

    // Get number of records in file
    fseek(database, 0L, SEEK_END);
    long fileLen = ftell(database);
    long records = fileLen / (schema->reclen);
    long seekLen = (records / 2);
    rewind(database);
    do
    {
        compareVal = 0;
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(print_string, 0, MAXINPUTLENGTH);
        fseek(database, seekLen * schema->reclen, SEEK_CUR);
        while (field != NULL)
        {
            fread(buffer, (size_t) field->length, 1, database);
            strncat(print_string, buffer, (size_t) field->length);
            strncat(print_string, ",", 1);
            field = field->next;
        }
        field = schema->fields->head;
        trimChars(print_string, ",");
        fprintf(output, "==> TRACE: %s\n", print_string);
        compareVal = strncmp(print_string, clause->compareVal, strlen(clause->compareVal));
        fseek(database, -1 * schema->reclen, SEEK_CUR);
        if (compareVal < 0)
        {
            seekLen = seekLen / 2;
            if (seekLen == 0) seekLen = 1;
        }
        else if (compareVal > 0)
        {
            seekLen = -1 * seekLen / 2;
            if (seekLen == 0) seekLen = -1;
        }
    } while (compareVal != 0 && iter < MAXBINSEARCH);
    do
    {
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(print_string, 0, MAXINPUTLENGTH);
        fseek(database, -2 * schema->reclen, SEEK_CUR);
        while (field != NULL)
        {
            fread(buffer, (size_t) field->length, 1, database);
            strncat(print_string, buffer, (size_t) field->length);
            strncat(print_string, ",", 1);
            field = field->next;
        }
        field = schema->fields->head;
        trimChars(print_string, ",");
        fprintf(output, "==> TRACE: %s\n", print_string);
        compareVal = strncmp(print_string, clause->compareVal, strlen(clause->compareVal));
    } while (compareVal == 0);

    do
    {
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(print_string, 0, MAXINPUTLENGTH);
        while (field != NULL)
        {
            fread(buffer, (size_t) field->length, 1, database);
            strncat(print_string, buffer, (size_t) field->length);
            strncat(print_string, ",", 1);
            field = field->next;
        }
        field = schema->fields->head;
        trimChars(print_string, ",");
        compareVal = strncmp(print_string, clause->compareVal, strlen(clause->compareVal));

        if (compareVal == 0 && strlen(print_string) > 0)
        {
            fprintf(output, "%s\n", print_string);
        }
    } while (compareVal == 0);
}

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################

/**
 * @brief getRecord finds all of the fields requested and prints them out
 * @param schema - reference to loaded table
 * @param selects - reference to linked list of fields to get
 */
void getRecord(_table *schema, linkedList *selects, FILE *output)
{
    // Initialize values
    FILE *database;
    char *buffer = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: BUFFER */
            *print_string = calloc(MAXINPUTLENGTH, 1), /** ALLOCATE: PRINT_STRING */
            *db_name = calloc(strlen(schema->tableFileName) + 5, 1);
    fieldNode *field = schema->fields->head;
    node *select = selects->head;
    strcat(db_name, "temp_");
    strcat(db_name, schema->tableFileName);
    if (access(db_name, F_OK) != -1)
    {
        database = fopen(db_name, "rb"); /** OPEN: DATABASE */
    }
    else
    {
        database = fopen(schema->tableFileName, "rb"); /** OPEN: DATABASE */
    }
    do
    {
        while (field != NULL)
        {
            fread(buffer, (size_t) field->length, 1, database);
            while (select != NULL)
            {
                size_t fieldNameLen = strlen(field->fieldName);
                size_t selectNameLen = strlen(select->field);

                size_t complen = (fieldNameLen > selectNameLen) ? fieldNameLen : selectNameLen;

                if (compareStrings(field->fieldName, select->field, complen, 0))
                {
                    strcpy(select->compareVal, buffer);
                    break;
                }
                select = select->next;
            }
            select = selects->head;
            field = field->next;
        }
        while (select != NULL)
        {
            strcat(print_string, select->compareVal);
            strcat(print_string, ",");
            memset(select->compareVal, 0, MAXINPUTLENGTH);
            select = select->next;
        }
        select = selects->head;
        trimwhitespace(print_string);
        trimChars(print_string, ",");
        if (strlen(print_string) > 0)
        {
            fprintf(output, "%s\n", print_string);
        }
        memset(buffer, 0, MAXINPUTLENGTH);
        memset(print_string, 0, MAXINPUTLENGTH);
        field = schema->fields->head;
    } while (!feof(database));
    fclose(database);  /** CLOSE: DATABASE */
    free(buffer); /** DEALLOCATE: BUFFER */
    free(print_string); /** DEALLOCATE: PRINT_STRING */
}

/**
 * @brief creates a list of fields to select from a table and whether a where clause was included, then calls the
 * appropriate function call.
 * @param buffer - pointer to stdin
 */
bool selectRecord(char *buffer)
{
    char *cmd = strtok(buffer, ", ");
    cmd = strtok(NULL, ", ");
    linkedList *fields = calloc(sizeof(linkedList), 1), /** ALLOCATE: FIELDS */
            *tables = calloc(sizeof(linkedList), 1), /** ALLOCATE: TABLES */
            *clauses = calloc(sizeof(linkedList), 1); /** ALLOCATE: CLAUSES */
    _table *schema = calloc(sizeof(schema), 1); /** ALLOCATE: SCHEMA */

    // Read in comma delimited fields and create linked list of fields to select.
    while (cmd != NULL)
    {
        addNode(fields, false, cmd, "", 0, false);
        cmd = strtok(NULL, ", ");
    }

    // Read in comma delimited tables and create linked list of tables to join.
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");
    cmd = strtok(NULL, ", ");
    while (cmd != NULL)
    {
        addNode(tables, false, cmd, "", 0, false);
        cmd = strtok(NULL, ", ");
    }

    // Read in where clauses and create linked list of wheres to join on.
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");
    if (strcmp(cmd, "WHERE") == 0)
    {
        // Initialize fields
        char *condition, *field;
        bool constant;
        while (!compareStrings(cmd, "END", 3, 0))
        {
            // Initialize fields
            condition = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: CONDITION */
            field = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: FIELD */
            constant = false;

            // Create field name and string to match for where clause
            cmd = strtok(NULL, " ");
            strncat(field, cmd, MAXINPUTLENGTH);
            cmd = strtok(NULL, " ");
            int compareCondition = 9;
            if (compareStrings(cmd, "=", 1, 0))
            {
                compareCondition = 0;
            }
            else if (compareStrings(cmd, ">=", 2, 0))
            {
                compareCondition = 2;
            }
            else if (compareStrings(cmd, "<=", 2, 0))
            {
                compareCondition = -2;
            }
            else if (compareStrings(cmd, ">", 1, 0))
            {
                compareCondition = 1;
            }
            else if (compareStrings(cmd, "<", 1, 0))
            {
                compareCondition = -1;
            }
            cmd = strtok(NULL, " ");
            if (compareStrings(cmd, "\"", 1, 0))
            {
                constant = true;
            }
            cmd = trimChars(cmd, "\"");
            strncat(condition, cmd, MAXINPUTLENGTH);
            addNode(clauses, false, field, condition, compareCondition, constant);
            free(field); /** DEALLOCATE: FIELD */
            free(condition); /** DEALLOCATE: CONDITION */
            fgets(buffer, MAXINPUTLENGTH - 1, stdin);
            trimwhitespace(buffer);
            printf("===> %s\n", buffer);
            cmd = strtok(buffer, " ");
        }
        // Read next line
        memset(buffer, 0, MAXINPUTLENGTH);
        if (tables->count > 1 && clauses->count > 0)
        {
            int temp_count = 0;
            while (tables->count > 1)
            {
                temp_count++;
                char *temp_name = calloc(4, 1);
                sprintf(temp_name, "tt_%d", temp_count);
                _table *join1 = calloc(sizeof(_table), 1), /** ALLOCATE: JOIN1 */
                        *join2 = calloc(sizeof(_table), 1); /** ALLOCATE: JOIN2 */

                // Load both schemas in to join
                strcpy(buffer, tables->head->field);
                loadSchema(join1, buffer);
                memset(buffer, 0, MAXINPUTLENGTH);
                strcpy(buffer, tables->head->next->field);
                loadSchema(join2, buffer);

                // Eliminate unnecessary records from each database first
                checkWhereLiteral(join1, tables->head, clauses);
                checkWhereLiteral(join2, tables->head->next, clauses);

                // Create temp table with join
                joinTable(join1, join2, clauses, temp_name);

                // Update Linked List
                popNode(tables);
                popNode(tables);
                addNode(tables, true, temp_name, "", 0, false);

                free(join1); /** DEALLOCATE: JOIN1 */
                free(join2); /** DEALLOCATE: JOIN2 */
            }
        }
        memset(buffer, 0, MAXINPUTLENGTH);
        strcpy(buffer, tables->head->field);
        loadSchema(schema, buffer);
        if (tables->count == 1 && clauses->count > 0 && schema->index == false)
        {
            checkWhereLiteral(schema, tables->head, clauses);
        }
    }
    // Pass in fields to read without where clause info
    memset(buffer, 0, MAXINPUTLENGTH);
    strcpy(buffer, tables->head->field);
    loadSchema(schema, buffer);
    if (clauses->count > 0 && schema->index == true)
    {
        getIndexedRecord(schema, fields, clauses, stdout);
    }
    else
    {
        getRecord(schema, fields, stdout);
    }
    free(fields); /** DEALLOCATE: FIELDS */
    free(tables); /** DEALLOCATE: TABLES */
    free(clauses); /** DEALLOCATE: CLAUSES */
}

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
                size_t bufLen = strlen(buffer);
                size_t compValLen = strlen(where->compareVal);
                size_t fCompLen = (fieldLen > whereLen) ? fieldLen : whereLen;
                size_t vCompLen = (bufLen > compValLen) ? bufLen : compValLen;
                if (where->constant == true &&
                    compareStrings(field->fieldName, where->field, fCompLen, 0) &&
                    !compareStrings(buffer, where->compareVal, vCompLen, where->conditional))
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

// #############################################################################
// ### MAIN FUNCTIONS
// #############################################################################
/**
 * @brief Reads input command from buffer and calls appropriate function
 * @param buffer - pointer to char array read from source
 */

void processCommand(char *buffer)
{
    char *cmd;
    char *tocomp1 = calloc(10, sizeof(char));
    char *tocomp2 = calloc(10, sizeof(char));
    char *tocomp3 = calloc(10, sizeof(char));
    strcat(tocomp1, "testing0");
    strcat(tocomp2, "testing1");
    strcat(tocomp3, "testing2");
    if (compareStrings(buffer, "CREATE TABLE", 12, 0))
    {
        cmd = strtok(buffer, " ");
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        createSchema(cmd, buffer, stdin, false, true);
    }
    else if (compareStrings(buffer, "CREATE INDEX", 12, 0))
    {
        createIndex(buffer, stdin);
    }
    else if (compareStrings(buffer, "INSERT", 6, 0))
    {
        char *temp = calloc(MAXINPUTLENGTH, 1);
        strncpy(temp, buffer, MAXINPUTLENGTH);
        cmd = strtok(buffer, " ");
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, " \n");
        _table *table = (_table *) calloc(sizeof(_table), 1);
        if (loadSchema(table, cmd))
        {
            // printSchema(table);
            loadDatabase(table, temp);
        }
        memset(table, 0, sizeof(_table));
        free(table);
        free(temp);
    }
    else if (compareStrings(buffer, "SELECT", 6, 0))
    {
        selectRecord(buffer);
    }
    else if (compareStrings(buffer, "DROP", 4, 0))
    {
        cmd = strtok(buffer, " ");
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        dropTable(cmd);
    }
    else if (compareStrings(buffer, "CLEAN ALL", 9, 0))
    {
        system("rm -f *.bin > /dev/null");
        system("rm -f *.schema > /dev/null");
        system("rm -f garbage* > /dev/null");
    }
    else if (compareStrings(buffer, "CLEAN TEMP", 10, 0))
    {
        system("rm -f tt* > /dev/null");
        system("rm -f temp_* > /dev/null");
        system("rm -f garbage* > /dev/null");
    }
}

int main()
{
    static char buffer[MAXINPUTLENGTH];
    memset(buffer, 0, MAXINPUTLENGTH);
    printf("Welcome!\n");
    char *status = fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    while (status != NULL)
    {
        trimwhitespace(buffer);
        if (strlen(buffer) < 3)
            break;
        printf("===> %s\n", buffer);
        processCommand(buffer);
        status = fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        system("rm -f tt* > /dev/null");
        system("rm -f temp_* > /dev/null");
        system("rm -f garbage* > /dev/null");
    }
    printf("Goodbye!\n");
    return 0;
}
