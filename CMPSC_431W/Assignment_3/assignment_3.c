#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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


bool createSchema()
{

}


bool loadSchema(struct _table *table)
{
    int field_number = 0;
    char *str_in = calloc(MAXINPUTLENGTH, sizeof(char));
    fgets(str_in, MAXINPUTLENGTH, stdin);
    char *current = strtok(str_in, " \n");
    current = strcat(current, strtok(NULL, " "));
    if (strcmp(current, "CREATETABLE") == 0)
    {
        printf("*** LOG: Loading table fields...\n");
        current = strtok(NULL, " \n");
        table->tableFileName = calloc(MAXLENOFFIELDNAMES, sizeof(char));
        strncpy(table->tableFileName, current, MAXLENOFFIELDNAMES);
        strcat(table->tableFileName, ".bin");
        printf("*** LOG: Table name is [%s]\n", table->tableFileName);
        table->reclen = 0;
    } else
    {
        return false;
    }
    do
    {
        current = strtok(str_in, " \n");
        if (strcmp(current, "ADD") == 0)
        {
            struct _field *current_field = &table->fields[field_number];
            table->fieldcount++;
            strncpy(current_field->fieldName, strtok(NULL, " \n"), MAXLENOFFIELDNAMES);
            strncpy(current_field->fieldType, strtok(NULL, " \n"), MAXLENOFFIELDTYPES);
            current_field->fieldLength = atoi(strtok(NULL, " \n"));
            table->reclen += current_field->fieldLength;
            printf("*** LOG: ADDING FIELD [%s] [%s] [%d]\n",
                   current_field->fieldName, current_field->fieldType, current_field->fieldLength);
            field_number++;
        }
        free(str_in);
        str_in = calloc(MAXINPUTLENGTH, sizeof(char));
        fgets(str_in, MAXINPUTLENGTH, stdin);
    } while (strncmp(str_in, "END", 3) != 0);
    free(str_in);
    printf("*** LOG: END OF CREATE TABLE\n");
    printf("*** LOG: %d Fields loaded\n", table->fieldcount);
    printf("*** LOG: Total record length is %d\n", table->reclen);
    return true;
}

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

    fgets(str_in, MAXINPUTLENGTH, stdin);
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
        fgets(str_in, MAXINPUTLENGTH, stdin);
    } while (str_in != NULL && strlen(str_in) > 11);
    printf("*** LOG: Closing file\n");
    fclose(database);
    return true;
}

// GET THE RECORD FROM THE FILE BY FSEEKING TO THE RIGHT SPOT AND READING IT
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


// DISPLAY THE CURRENT RECORD USING THE ASSOCIATED FIELD NAMES
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

void trimwhitespace(char *to_trim)
{
    while (*to_trim == ' ')
    {
        
        to_trim++;
        *to_trim = 0;
    }
    while (*to_trim != '\0')
    {
        to_trim++;
    }
    while (*to_trim == ' ')
    {
        to_trim--;
        *to_trim = 0;
    }
    printf("%s\n", to_trim);
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
            break; // not a real command, CR/LF, extra line, etc.
        printf("===> %s\n", buffer);
        //processCommand(buffer);
        status = fgets(buffer, MAXINPUTLENGTH - 1, stdin);
    }
    printf("Goodbye!\n");
    return 0;
}
