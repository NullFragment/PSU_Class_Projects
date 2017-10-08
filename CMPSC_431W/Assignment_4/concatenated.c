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

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################


bool getRecord(int recnum, char *record, struct _table *table)
{
    char *filename = table->tableFileName;
    FILE *database;
    database = fopen(filename, "rb");
    fseek(database, recnum * (table->reclen), SEEK_SET);
    fread(record, (size_t) table->reclen, 1, database);
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
    int *field_numbers = calloc((unsigned) schema->fieldcount, sizeof(int));
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
    fread(buffer, (unsigned) schema->fields[0].fieldLength, 1, table);
    while (!feof(table))
    {
        for (int j = 0; j < field_counter; j++)
        {
            if (field_numbers[j] == 0)
            {
                printf("%s ", buffer);
            }
        }
        for (int i = 1; i < schema->fieldcount; i++)
        {
            fread(buffer, (unsigned) schema->fields[i].fieldLength, 1, table);
            for (int j = 0; j < field_counter; j++)
            {
                if (field_numbers[j] == i)
                {
                    printf("%s ", buffer);
                }
            }
            memset(buffer, 0, MAXINPUTLENGTH);
        }
        printf("\n");
        fread(buffer, (unsigned) schema->fields[0].fieldLength, 1, table);
    }
    fclose(table);
}

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
    char *file_name = calloc(1, MAXLENOFFIELDNAMES + 8); /** ALLOCATE: FILE NAME */
    strcat(file_name, schema_name);
    strcat(file_name, ".schema");

    // Exit out if schema file does not exist
    if (access(file_name, F_OK) == -1) return false;

    FILE *schema = fopen(file_name, "rb"); /** OPEN FILE: SCHEMA */

    // Initialize number of fields counter and buffer string
    int field_number = 0;
    char *str_in = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: STR IN */
    fread(str_in, MAXINPUTLENGTH - 1, 1, schema);

    // Initialize table metadata
    table->tableFileName = calloc(MAXLENOFFIELDNAMES, sizeof(char));
    strncpy(table->tableFileName, schema_name, MAXLENOFFIELDNAMES);
    strcat(table->tableFileName, ".bin");
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
            field_number++;
        }
        free(str_in);
        str_in = calloc(MAXINPUTLENGTH, sizeof(char));
        fread(str_in, MAXINPUTLENGTH - 1, 1, schema);
    } while (strlen(str_in) > 3);
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
bool createSchema(char *schema_name, char *buffer)
{
    // Allocate memory for and create filename
    char *file_name = calloc(1, MAXLENOFFIELDNAMES + 8); /** ALLOCATE: FILE NAME */
    strcat(file_name, schema_name);
    strcat(file_name, ".schema");

//    if (access(file_name, F_OK) == -1) // UNCOMMENT TO NOT OVERWRITE SCHEMA FILES
//    {

    FILE *schema = fopen(file_name, "wb+"); /** OPEN FILE: SCHEMA */
    memset(buffer, 0, MAXINPUTLENGTH);
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);

    // Start reading in schema structure and saving to file
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    while (strncmp(buffer, "END", 3) != 0 && buffer != NULL)
    {
        fwrite(buffer, MAXINPUTLENGTH - 1, 1, schema);
        fwrite("\n", 1, 1, schema);
        memset(buffer, 0, MAXINPUTLENGTH);
        fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);
    }
    fwrite("END\n", 4, 1, schema);
    fclose(schema); /** CLOSE FILE: SCHEMA */
    free(file_name); /** DEALLOCATE: FILE NAME */
//    } // UNCOMMENT TO NOT OVERWRITE SCHEMA FILES

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

// #############################################################################
// ### DATABASE FUNCTIONS
// #############################################################################
/**
 * @brief Saves data into a .schema file given a table structure for reference
 * @param table - pointer to table structure generated with loadSchema
 * @return returns true if function completes.
 */
bool loadDatabase(struct _table *table)
{
    // Initialize values
    char *str_in, *record, *current,
            *filename = table->tableFileName; /** ALLOCATE: FILENAME */
    int record_length = table->reclen,
            rec_loc = 0;
    FILE *database;

    database = fopen(filename, "wb+"); /** OPEN FILE: DATABASE */
    record = calloc(1, (size_t) record_length); /** ALLOCATE: RECORD */
    str_in = calloc(MAXINPUTLENGTH, sizeof(char)); /** ALLOCATE: STR IN */
    fgets(str_in, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(str_in);
    printf("===> %s\n", str_in);
    do
    {
        current = strtok(str_in, ",\n");
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
        fwrite(record, record_length, 1, database);
        fwrite("\n", 1, 1, database);
        // Reset values to empty
        rec_loc = 0;
        free(str_in);
        free(record);
        record = calloc(1, (size_t) record_length);
        str_in = calloc(MAXINPUTLENGTH, sizeof(char));

        // Read in next record
        fgets(str_in, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(str_in);
        printf("===> %s\n", str_in);
    } while (strncmp(str_in, "END", 3) != 0);
    fclose(database); /** CLOSE FILE: DATABASE */
    free(str_in); /** DEALLOCATE: STR IN */
    free(record); /** DEALLOCATE: RECORD */
    return true;
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
    char *cmd = strtok(buffer, " ");
    if (strcmp(cmd, "CREATE") == 0)
    {
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        createSchema(cmd, buffer);
    } else if (strcmp(cmd, "LOAD") == 0)
    {
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, " \n");
        struct _table table;
        if (loadSchema(&table, cmd))
        {
            //printSchema(&table);
            loadDatabase(&table);
        }
    } else if (strcmp(cmd, "SELECT") == 0)
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
            loadSchema(&table, cmd);
            selectRecord(&table, fields);
        }

    }
}

int main()
{
    int x = 0;
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
        x = 0;

    }
    printf("Goodbye!\n");
    return 0;
}
