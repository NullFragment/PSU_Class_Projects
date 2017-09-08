#define DATABASENAME "database.bin"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

struct _record
{
    char id[50];
    char lname[50];
    char fname[50];
    char mname[50];
    char suffix[50];
    char bday[50];
    char gender[50];
    char ss_num[50];
    char address1[50];
    char address2[50];
    char zip[50];
    char maiden[50];
    char mrn[50];
    char city[50];
    char state[50];
    char phone1[50];
    char phone2[50];
    char email[50];
    char alias[50];
};

_Bool loadDatabase(char file[255])
{
    FILE *database;
    struct _record read_in;
    char *fields, str_in[2000];
    printf("*** LOG: Loading database from input ***\n");
    database = fopen(file, "wb");
    fclose(database);
    database = fopen(file, "ab");

    fgets(str_in,2000,stdin);
    while (strlen(str_in) > 11) {
        printf("*** LOG: Parsing input data starting with [%.20s]\n", str_in);
        fields = strtok(str_in, ",");
        strncpy(read_in.id, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.lname, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.fname, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.mname, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.suffix, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.bday, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.gender, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.ss_num, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.address1, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.address2, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.zip, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.maiden, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.mrn, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.city, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.state, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.phone1, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.phone2, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.email, fields, 49);
        fields = strtok(NULL, ",");
        strncpy(read_in.alias, fields, 49);
        fwrite(&read_in, sizeof(struct _record), 1, database);
        printf("*** LOG: Appending record to database ***\n");
        fgets(str_in,2000,stdin);
    }
    fclose(database);
    return true;
}

_Bool getRecord(char file[255], int record_number, struct _record *record)
{
    FILE *database;
    printf("*** LOG: Getting record %d from the database ***\n", record_number);
    database = fopen(file, "r+b");
    fseek(database, record_number * sizeof(struct _record), SEEK_SET);
    fread(record, sizeof(struct _record), 1, database);
    fclose(database);
    return true;
}

_Bool showRecord(struct _record *record)
{
    printf("---------------------------------\n");
    printf("ID: %s\n", record->id);
    printf("Last name: %s\n", record->lname);
    printf("First name: %s\n", record->fname);
    printf("Middle name: %s\n", record->mname);
    printf("Suffix: %s\n", record->suffix);
    printf("Birth date: %s\n", record->bday);
    printf("Gender: %s\n", record->gender);
    printf("SS Num: %s\n", record->ss_num);
    printf("Address 1: %s\n", record->address1);
    printf("Address 2: %s\n", record->address2);
    printf("Zip: %s\n", record->zip);
    printf("Maiden: %s\n", record->maiden);
    printf("MRN: %s\n", record->mrn);
    printf("City: %s\n", record->city);
    printf("State: %s\n", record->state);
    printf("Phone 1: %s\n", record->phone1);
    printf("Phone 2: %s\n", record->phone2);
    printf("Email: %s\n", record->email);
    printf("Alias: %s\n", record->alias);
    return true;
}

_Bool changeRecord(char file[255], int record_number, char field[10], char new_string[49])
{
    FILE *database;
    struct _record temp;
    printf("*** LOG: Changing field %s in record %d to [%s] ***\n", field, record_number, new_string);
    printf("*** LOG: Getting record %d from the database ***\n", record_number);
    database = fopen(file, "r+b");
    fseek(database, record_number * sizeof(struct _record), SEEK_SET);
    fread(&temp, sizeof(struct _record), 1, database);

    if(strcmp(field, "id") == 0)
    {
        strncpy(temp.id, new_string, 49);
    }
    else if(strcmp(field, "lname") == 0)
    {
        strncpy(temp.lname, new_string, 49);
    }
    else if(strcmp(field, "fname") == 0)
    {
        strncpy(temp.fname, new_string, 49);
    }
    else if(strcmp(field, "mname") == 0)
    {
        strncpy(temp.mname, new_string, 49);
    }
    else if(strcmp(field, "suffix") == 0)
    {
        strncpy(temp.suffix, new_string, 49);
    }
    else if(strcmp(field, "bday") == 0)
    {
        strncpy(temp.bday, new_string, 49);
    }
    else if(strcmp(field, "gender") == 0)
    {
        strncpy(temp.gender, new_string, 49);
    }
    else if(strcmp(field, "ss_num") == 0)
    {
        strncpy(temp.ss_num, new_string, 49);
    }
    else if(strcmp(field, "address1") == 0)
    {
        strncpy(temp.address1, new_string, 49);
    }
    else if(strcmp(field, "address2") == 0)
    {
        strncpy(temp.address2, new_string, 49);
    }
    else if(strcmp(field, "zip") == 0)
    {
        strncpy(temp.zip, new_string, 49);
    }
    else if(strcmp(field, "maiden") == 0)
    {
        strncpy(temp.maiden, new_string, 49);
    }
    else if(strcmp(field, "mrn") == 0)
    {
        strncpy(temp.mrn, new_string, 49);
    }
    else if(strcmp(field, "city") == 0)
    {
        strncpy(temp.city, new_string, 49);
    }
    else if(strcmp(field, "state") == 0)
    {
        strncpy(temp.state, new_string, 49);
    }
    else if(strcmp(field, "phone1") == 0)
    {
        strncpy(temp.phone1, new_string, 49);
    }
    else if(strcmp(field, "phone2") == 0)
    {
        strncpy(temp.phone2, new_string, 49);
    }
    else if(strcmp(field, "email") == 0)
    {
        strncpy(temp.email, new_string, 49);
    }
    else if(strcmp(field, "alias") == 0)
    {
        strncpy(temp.alias, new_string, 49);
    }
    printf("*** LOG: Updating record %d in the database ***\n", record_number);
    fseek(database, record_number * sizeof(struct _record), SEEK_SET);
    fwrite(&temp, sizeof(struct _record), 1, database);
    fclose(database);
    return true;
}
