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

/**
 * @brief Trims quotes from a given character array
 * @param to_trim - pointer to array to trim whitespace from
 */
char *trimQuotes(char *to_trim)
{
    char *j;
    while (strncmp(to_trim, "\"", 1) == 0)
    {
        to_trim++;
    }
    size_t length = strlen(to_trim);
    j = to_trim + length - 1;
    while (strcmp(j, "\"") == 0)
    {
        *j = 0;
        j--;
    }
    return to_trim;
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

bool loadSchema(struct _table *table, char *buffer)
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
        fwrite(record, record_length - 1, 1, database);
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
    //free(current); /** DEALLOCATE: CURRENT */
    //free(database); /** DEALLOCATE: DATABASE*/
    //free(filename); /** DEALLOCATE: RECORD */
    free(str_in); /** DEALLOCATE: STR IN */
    free(record); /** DEALLOCATE: RECORD */
    return true;
}

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################

/**
 * @brief finds all records in a given populated table
 * @param schema - pointer to loaded schema
 * @param fields - pointer to comma separated string of fields
 * @param to_match - pointer to character of field to compare against for where clause
 * @param condition - pointer to value to compare with for where clause
 */
void getRecord(struct _table *schema, char *fields, char *to_match, char *condition)
{
    // Initialize values
    char *buffer = calloc(MAXINPUTLENGTH, 1);
    int field_counter = 0, matching_field = -1;
    bool where_check = false, print_flag = false;
    int *field_numbers = calloc((unsigned) schema->fieldcount, sizeof(int));
    char *field = strtok(fields, ",");

    // Determine whether or not to check where clause
    if (strlen(to_match) > 0)
    {
        where_check = true;
    }

    // Find all matching fields and create an array of their indices.
    while (field != NULL)
    {
        for (int i = 0; i < schema->fieldcount; i++)
        {
            if (strcmp(schema->fields[i].fieldName, field) == 0)
            {
                field_numbers[field_counter] = i;
                field_counter++;
            }
            if (where_check == true && matching_field < 0 && strcmp(schema->fields[i].fieldName, to_match) == 0)
            {
                matching_field = i;
            }
        }
        field = strtok(NULL, ",");
    }

    // Open schema file and search through all records for wanted information
    FILE *table = fopen(schema->tableFileName, "rb");
    strtok(buffer, " \n\0");
    while (!feof(table))
    {
        char *to_print = calloc(sizeof(char), MAXINPUTLENGTH);
        for (int i = 0; i < schema->fieldcount; i++)
        {
            fread(buffer, (unsigned) schema->fields[i].fieldLength, 1, table);
            trimwhitespace(buffer);
            if (strlen(buffer) == 0) break;
            for (int j = 0; j < field_counter; j++)
            {
                if (field_numbers[j] == i)
                {
                    trimwhitespace(buffer);
                    strcat(to_print, buffer);
                    strcat(to_print, " ");
                }
            }
            if (where_check == true && i == matching_field && strcmp(buffer, condition) == 0)
            {
                print_flag = true;
            }
            memset(buffer, 0, MAXINPUTLENGTH);
        }

        if (strlen(to_print) > 0 && (print_flag == true || where_check == false))
        {
            printf("%s\n", to_print);
            print_flag = false;
        }
        free(to_print);
    }
    fclose(table);
}


/**
 * @brief creates a list of fields to select from a table and whether a where clause was included, then calls the
 * appropriate function call.
 * @param buffer - pointer to stdin
 */
bool selectRecord(char *buffer)
{
    char *cmd = strtok(NULL, ", ");
    char *fields = calloc(MAXINPUTLENGTH, 1);
    struct _table table;

    // Read in comma delimited fields and reconstruct search field array.
    while (cmd != NULL)
    {
        strncat(fields, cmd, MAXINPUTLENGTH - strlen(fields) - 1);
        strcat(fields, ",");
        cmd = strtok(NULL, ", ");
    }

    // Read next line
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");

    // Load table if it exists, if not, break early
    if (strcmp(cmd, "FROM") == 0)
    {
        cmd = strtok(NULL, " \n");
        if (!loadSchema(&table, cmd)) return false;
    }

    // Read next line
    fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    trimwhitespace(buffer);
    printf("===> %s\n", buffer);
    cmd = strtok(buffer, ", ");

    if (strcmp(cmd, "WHERE") == 0)
    {
        // Initialize fields
        char *condition = calloc(MAXLENOFFIELDNAMES, sizeof(char)),
                *field = calloc(MAXINPUTLENGTH, sizeof(char));

        // Create field name and string to match for where clause
        cmd = strtok(NULL, " ");
        strncat(field, cmd, MAXLENOFFIELDNAMES);
        cmd = strtok(NULL, " =");
        cmd = trimQuotes(cmd);
        strncat(condition, cmd, MAXINPUTLENGTH);

        // Read next line
        fgets(buffer, MAXINPUTLENGTH - 1, stdin);
        trimwhitespace(buffer);
        printf("===> %s\n", buffer);

        getRecord(&table, fields, field, condition);
    }
    else if (strcmp(cmd, "WHERE") != 0)
    {
        // Pass in fields to read without where clause info
        getRecord(&table, fields, "", "");
    }
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
    }
    else if (strcmp(cmd, "LOAD") == 0)
    {
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, " \n");
        struct _table *table = (struct _table *) malloc(sizeof(struct _table));
        if (loadSchema(table, cmd))
        {
            // printSchema(table);
            loadDatabase(table);
        }
        memset(table, 0, sizeof(struct _table));
        free(table);
    }
    else if (strcmp(cmd, "SELECT") == 0)
    {
        selectRecord(buffer);
    }
    else if (strcmp(cmd, "DROP") == 0)
    {
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        dropTable(cmd);
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

