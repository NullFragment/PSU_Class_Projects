#include "utils.h"
#include "functions_database.h"
#include "functions_schema.h"
#include "functions_records.h"

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
