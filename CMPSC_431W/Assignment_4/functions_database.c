#include "utils.h"
#include "functions_database.h"

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
    free(current); /** DEALLOCATE: CURRENT */
    free(database); /** DEALLOCATE: DATABASE*/
    free(filename); /** DEALLOCATE: RECORD */
    free(str_in); /** DEALLOCATE: STR IN */
    free(record); /** DEALLOCATE: RECORD */
    return true;
}