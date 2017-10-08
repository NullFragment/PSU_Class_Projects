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
        struct _table *table = (struct _table*)malloc(sizeof(struct _table));
        if (loadSchema(table, cmd))
        {
            //printSchema(&table);
            loadDatabase(table);
        }
        memset(table, 0, sizeof(struct _table));
        free(table);
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
