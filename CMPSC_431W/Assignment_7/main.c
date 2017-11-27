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
    if (strncmp(buffer, "CREATE TABLE", 12) == 0)
    {
        cmd = strtok(buffer, " ");
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        createSchema(cmd, buffer, stdin, false, true);
    }
    if (strncmp(buffer, "CREATE INDEX", 12) == 0)
    {
        createIndex(buffer, stdin);
    }
    else if (strncmp(buffer, "INSERT", 6) == 0)
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
    else if (strncmp(buffer, "SELECT", 6) == 0)
    {
        selectRecord(buffer);
    }
    else if (strncmp(buffer, "DROP", 4) == 0)
    {
        cmd = strtok(buffer, " ");
        cmd = strtok(NULL, " ");
        cmd = strtok(NULL, "\n");
        dropTable(cmd);
    }
    else if (strncmp(buffer, "CLEAN ALL", 9) == 0)
    {
        system("rm *.bin");
        system("rm *.schema");
        system("rm garbage*");
    }
    else if (strncmp(buffer, "CLEAN TEMP", 10) == 0)
    {
        system("rm tt*");
        system("rm temp_*");
        system("rm garbage*");
    }
}

int main()
{
    //system("rm -rf *.bin > nul");
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
