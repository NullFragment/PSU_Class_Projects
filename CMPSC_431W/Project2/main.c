// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define MAXFIELDS 100 // for now
#define MAXINPUTLENGTH 5000
#define MAXLENOFFIELDNAMES 50
#define MAXLENOFFIELDTYPES 50
struct _field {
    char fieldName[MAXLENOFFIELDNAMES];
    char fieldType[MAXLENOFFIELDTYPES];
    int fieldLength;
};
struct _table {
    char *tableFileName;
    int reclen;
    int fieldcount;
    struct _field fields[MAXFIELDS];
};
typedef enum{false, true} bool;
// READ FROM STDIN AND WRITE THE RECORDS TO THE DATABASE FILE
// THE DATABASE FILE MUST BE CALLED [table name].bin <-- where "table name" 
// is the table named in the schema
// BE CERTAIN TO WRITE THESE IN THE FORMAT AS SPECIFIED BY THE SCHEMA
// MAKE SURE THAT THE DATA LENGTHS DO NOT TRASH YOUR FILE!
// Note that I made the design decision to store the string
// null terminators in the file. 
bool loadDatabase(struct _table *table) {
}
// READ THE DATA FROM STDIN AS THE DESIGN OF THE DATABASE TABLE
// LOAD "table" WITH THE APPROPRIATE INFO
bool loadSchema(struct _table *table) {
}
// GET THE RECORD FROM THE FILE BY FSEEKING TO THE RIGHT SPOT AND READING IT
bool getRecord(int recnum, char *record, struct _table *table){
}
// DISPLAY THE CURRENT RECORD USING THE ASSOCIATED FIELD NAMES
void showRecord(struct _field *fields, char *record, int fieldcount){
}
int main() {
    struct _table table;
    loadSchema(&table);
    loadDatabase(&table);
    char *record = calloc(1, table.reclen);
    if (record == NULL)
        printf("\n\n**** ERROR OUT OF MEMORY ***\n\n");
    else {
        if (getRecord(4, record, &table))
            showRecord(table.fields, record, table.fieldcount);
        if (getRecord(10, record, &table))
            showRecord(table.fields, record, table.fieldcount);
    }
    return 0;
}

