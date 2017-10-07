#include "utils.h"
#include "functions_records.h"

// #############################################################################
// ### RECORD FUNCTIONS
// #############################################################################


bool getRecord(int recnum, char *record, struct _table *table)
{
    char *filename = table->tableFileName;
    FILE *database;
    database = fopen(filename, "rb");
    fseek(database, recnum * (table->reclen), SEEK_SET);
    fread(record, (size_t)table->reclen, 1, database);
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
