#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#define MAXFIELDS 100 // for now
#define MAXINPUTLENGTH 5000
#define MAXLENOFFIELDNAMES 50
#define MAXLENOFFIELDTYPES 50

struct _field
{
    char fieldName[MAXLENOFFIELDNAMES];
    char fieldType[MAXLENOFFIELDTYPES];
    int fieldLength;
};

struct _table
{
    char *tableFileName;
    int reclen;
    int fieldcount;
    struct _field fields[MAXFIELDS];
};

typedef enum
{
    false, true
} bool;

// #############################################################################
// ### UTILITY FUNCTIONS
// #############################################################################


void trimwhitespace(char *to_trim)
{
    char *j = to_trim;
    while (isspace(*j))
    {
        j++;
    }
    size_t length = strlen(j);
    memmove(to_trim, j, length);
    j = to_trim + length - 1;
    while (isspace(*j))
    {
        *j = 0;
        j--;
    }
}


// #############################################################################
// ### SCHEMA FUNCTIONS
// #############################################################################

// LOAD SCHEMA
// *table - structure of table to create
// *schema_name - string that will be used to create filenames
// logging - if true, turns on print statements
// Function parses through a .schema file (if it exists) and assigns all schema
// fields to the given table struct.

bool loadSchema(struct _table *table, char *schema_name, bool logging)
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
    if (logging) printf("*** LOG: Loading table fields...\n");
    table->tableFileName = calloc(MAXLENOFFIELDNAMES, sizeof(char));
    strncpy(table->tableFileName, schema_name, MAXLENOFFIELDNAMES);
    strcat(table->tableFileName, ".bin");
    if (logging) printf("*** LOG: Table data name is [%s]\n", table->tableFileName);
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
    if (logging) printf("*** LOG: Table schema name is [%s]\n", filename);
    if (logging) printf("*** LOG: END OF CREATE TABLE\n");
    //printf("*** LOG: %d Fields loaded\n", table->fieldcount);
    //printf("*** LOG: Total record length is %d\n", table->reclen);
    return true;
}


// CREATE SCHEMA
// *file_name - takes name of file to be used excluding file extension
// *buffer - pointer to buffer for stdin
// Function takes the input, and simply saves schema creation "ADD"
// SQL-like commands to a file with a .schema extension.

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
    loadSchema(&table, schema_name, true);
    /*
    // UNCOMMENT TO NOT OVERWRITE SCHEMA FILES
    }
    */
}

// Parses through a given schema file
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

// #############################################################################
// ### DATABASE FUNCTIONS
// #############################################################################

bool loadDatabase(struct _table *table)
{
    char *str_in,
            *record,
            *current,
            *filename = table->tableFileName;
    int record_length = table->reclen,
            rec_loc = 0;
    FILE *database;

    database = fopen(filename, "wb+");
    record = calloc(1, record_length);
    str_in = calloc(MAXINPUTLENGTH, sizeof(char));

    fgets(str_in, MAXINPUTLENGTH - 1, stdin);
    printf("*** LOG: Loading database from input ***\n");
    do
    {
        printf("*** LOG: Loading input data starting with [%.20s]\n", str_in);
        current = strtok(str_in, ",\n");
        for (int i = 0; i < table->fieldcount; i++)
        {
            int f_length = table->fields[i].fieldLength;
            if (strlen(current) > f_length)
            {
                printf("*** WARNING: Data in field %s is being truncated ***\n", table->fields[i].fieldName);
            }
            strncat(&record[rec_loc], current, f_length - 1);
            rec_loc += f_length;
            current = strtok(NULL, ",\n");
        }
        rec_loc = 0;
        fwrite(record, record_length, 1, database);
        free(str_in);
        free(record);
        str_in = calloc(MAXINPUTLENGTH, sizeof(char));
        record = calloc(1, record_length);
        fgets(str_in, MAXINPUTLENGTH - 1, stdin);
    } while (str_in != NULL && strlen(str_in) > 11);
    printf("*** LOG: Closing file\n");
    fclose(database);
    return true;
}

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################


bool getRecord(int recnum, char *record, struct _table *table)
{
    char *filename = table->tableFileName;
    FILE *database;
    printf("*** LOG: Getting record %d from the database ***\n", recnum);
    database = fopen(filename, "rb");
    fseek(database, recnum * (table->reclen), SEEK_SET);
    fread(record, table->reclen, 1, database);
    fclose(database);
    return true;
}


void showRecord(struct _field *fields, char *record, int fieldcount)
{
    int rec_loc = 0;
    printf("----------- RECORD --------------\n");
    for (int i = 0; i < fieldcount; i++)
    {
        printf("--- %s: [%s]\n", fields[i].fieldName, &record[rec_loc]);
        rec_loc += fields[i].fieldLength;
    }
}


void selectRecord(struct _table *schema, char *fields)
{
    // Initialize values
    char *buffer = calloc(MAXINPUTLENGTH, 1);
    int field_counter = 0;
    int *field_numbers = calloc((uint)schema->fieldcount, sizeof(int));
    char *field = strtok(fields, ",");
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
        }
        field = strtok(NULL, ",");
    }

    // Open schema file and search through all records for wanted information
    FILE *table = fopen(schema->tableFileName, "rb");
    strtok(buffer, " \n\0");
    fread(buffer, (uint) schema->fields[0].fieldLength, 1, table);
    while (!feof(table))
    {
        for(int j = 0; j < field_counter; j++)
        {
            if(field_numbers[j] == 0)
            {
                printf("%s ", buffer);
            }
        }
        for(int i = 1; i < schema->fieldcount; i++)
        {
            fread(buffer, (uint) schema->fields[i].fieldLength, 1, table);
            for(int j = 0; j < field_counter; j++)
            {
                if(field_numbers[j] == i)
                {
                    printf("%s ", buffer);
                }
            }
        }
        printf("\n");
        fread(buffer, (uint) schema->fields[0].fieldLength, 1, table);
    }
    fclose(table);
}


void processCommand(char *buffer)
{
    char *cmd = strtok(buffer, " ");
    if (strcmp(cmd, "CREATE") == 0)
    {
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        createSchema(cmd, buffer);
    }
    else if (strcmp(cmd, "LOAD") == 0)
    {
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        struct _table table;
        if (loadSchema(&table, cmd, false))
        {
            printSchema(&table);
            loadDatabase(&table);
        }
    }
    else if (strcmp(cmd, "SELECT") == 0)
    {
        cmd = strtok(NULL, ", ");
        char *fields = calloc(MAXINPUTLENGTH, 1);
        while (strcmp(cmd, "FROM") != 0 && cmd != NULL)
        {
            strncat(fields, cmd, MAXINPUTLENGTH - strlen(fields) - 1);
            strcat(fields, ",");
            cmd = strtok(NULL, ", ");
        }
        if (strcmp(cmd, "FROM") == 0)
        {
            cmd = strtok(NULL, " \n");
            struct _table table;
            loadSchema(&table, cmd, false);
            selectRecord(&table, fields);
        }

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
        if (strlen(buffer) < 5)
            break;
        printf("===> %s\n", buffer);
        processCommand(buffer);
        status = fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    }
    printf("Goodbye!\n");
    return 0;
}
