#include "utils.h"
#include "functions_database.h"

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
    record = calloc(1, (size_t) record_length);
    str_in = calloc(MAXINPUTLENGTH, sizeof(char));
    fgets(str_in, MAXINPUTLENGTH - 1, stdin);
    do
    {
        current = strtok(str_in, ",\n");
        for (int i = 0; i < table->fieldcount; i++)
        {
            int f_length = table->fields[i].fieldLength;
            if (strlen(current) > f_length)
            {
                printf("*** WARNING: Data in field %s is being truncated ***\n", table->fields[i].fieldName);
            }
            strncat(&record[rec_loc], current, (size_t)(f_length - 1));
            rec_loc += f_length;
            current = strtok(NULL, ",\n");
        }
        rec_loc = 0;
        fwrite(record, (size_t)record_length, 1, database);
        free(str_in);
        free(record);
        str_in = calloc(MAXINPUTLENGTH, sizeof(char));
        record = calloc(1, (size_t)record_length);
        fgets(str_in, MAXINPUTLENGTH - 1, stdin);
    } while (str_in != NULL && strlen(str_in) > 11);
    printf("*** LOG: Closing file\n");
    fclose(database);
    return true;
}