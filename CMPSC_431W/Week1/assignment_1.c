#define DATABASENAME "database.bin"

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

struct _record
{
    char id[20];
    char lname[30];
    char fname[30];
    char mname[30];
    char suffix[10];
    char bday[10];
    char gender[8];
    char ss_num[11];
    char address1[30];
    char address2[30];
    char zip[10];
    char maiden[30];
    char mrn[30];
    char city[30];
    char state[5];
    char phone1[12];
    char phone2[12];
    char email[30];
    char alias[30];
};

_Bool loadDatabase(char file[255], char input[450])
{
    FILE *database;
    struct _record read_in;
    char *fields;
    fields = strtok(input, ",");

    database = fopen(file, "wb");
    fclose(database);

    database = fopen(file, "ab");
    while (fields != NULL) {
        strncpy(read_in.id, fields, 20);
        fields = strtok(NULL, ",");
        strncpy(read_in.lname, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.fname, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.mname, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.suffix, fields, 10);
        fields = strtok(NULL, ",");
        strncpy(read_in.bday, fields, 10);
        fields = strtok(NULL, ",");
        strncpy(read_in.gender, fields, 8);
        fields = strtok(NULL, ",");
        strncpy(read_in.ss_num, fields, 11);
        fields = strtok(NULL, ",");
        strncpy(read_in.address1, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.address2, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.zip, fields, 10);
        fields = strtok(NULL, ",");
        strncpy(read_in.maiden, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.mrn, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.city, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.state, fields, 5);
        fields = strtok(NULL, ",");
        strncpy(read_in.phone1, fields, 12);
        fields = strtok(NULL, ",");
        strncpy(read_in.phone2, fields, 12);
        fields = strtok(NULL, ",");
        strncpy(read_in.email, fields, 30);
        fields = strtok(NULL, ",");
        strncpy(read_in.alias, fields, 30);
        fwrite(&read_in, sizeof(struct _record), 1, database);
        fields = strtok(NULL, ",");
    }
    fclose(database);
    return true;
}

_Bool getRecord(char file[255], int record_number, struct _record *record)
{
    FILE *database;
    database = fopen(file, "r+b");
    fseek(database, record_number * sizeof(struct _record), SEEK_SET);
    fread(record, sizeof(struct _record), 1, database);
    fclose(database);
    return true;
}

_Bool showRecord(struct _record *record)
{
    printf("ID: %s \n", record->id);
    printf("Last Name: %s \n", record->lname);
    printf("First Name: %s \n", record->fname);
    printf("Middle Name: %s \n", record->mname);
    printf("Suffix: %s \n", record->suffix);
    printf("Birth Date: %s \n", record->bday);
    printf("Gender: %s \n", record->gender);
    printf("Social Security: %s \n", record->ss_num);
    printf("Address 1: %s \n", record->address1);
    printf("Address 2: %s \n", record->address2);
    printf("ZIP Code: %s \n", record->zip);
    printf("Maiden Name: %s \n", record->maiden);
    printf("MRN: %s \n", record->mrn);
    printf("City: %s \n", record->city);
    printf("State: %s \n", record->state);
    printf("Phone 1: %s \n", record->phone1);
    printf("Phone 2: %s \n", record->phone2);
    printf("E-Mail: %s \n", record->email);
    printf("Alias: %s \n", record->alias);
    return true;
}

_Bool changeRecord(char file[255], int record_number, char field[8], char new_string[40])
{
    FILE *database;
    struct _record temp;
    database = fopen(file, "r+b");
    fseek(database, record_number * sizeof(struct _record), SEEK_SET);
    fread(&temp, sizeof(struct _record), 1, database);

    if(strcmp(field, "id") == 0)
    {
        strncpy(temp.id, new_string, 20);
    }
    else if(strcmp(field, "lname") == 0)
    {
        strncpy(temp.lname, new_string, 30);
    }
    else if(strcmp(field, "fname") == 0)
    {
        strncpy(temp.fname, new_string, 30);
    }
    else if(strcmp(field, "mname") == 0)
    {
        strncpy(temp.mname, new_string, 30);
    }
    else if(strcmp(field, "suffix") == 0)
    {
        strncpy(temp.suffix, new_string, 10);
    }
    else if(strcmp(field, "bday") == 0)
    {
        strncpy(temp.bday, new_string, 10);
    }
    else if(strcmp(field, "gender") == 0)
    {
        strncpy(temp.gender, new_string, 8);
    }
    else if(strcmp(field, "ss_num") == 0)
    {
        strncpy(temp.ss_num, new_string, 11);
    }
    else if(strcmp(field, "address1") == 0)
    {
        strncpy(temp.address1, new_string, 30);
    }
    else if(strcmp(field, "address2") == 0)
    {
        strncpy(temp.address2, new_string, 30);
    }
    else if(strcmp(field, "zip") == 0)
    {
        strncpy(temp.zip, new_string, 10);
    }
    else if(strcmp(field, "maiden") == 0)
    {
        strncpy(temp.maiden, new_string, 30);
    }
    else if(strcmp(field, "mrn") == 0)
    {
        strncpy(temp.mrn, new_string, 30);
    }
    else if(strcmp(field, "city") == 0)
    {
        strncpy(temp.city, new_string, 30);
    }
    else if(strcmp(field, "state") == 0)
    {
        strncpy(temp.state, new_string, 5);
    }
    else if(strcmp(field, "phone1") == 0)
    {
        strncpy(temp.phone1, new_string, 12);
    }
    else if(strcmp(field, "phone2") == 0)
    {
        strncpy(temp.phone2, new_string, 12);
    }
    else if(strcmp(field, "email") == 0)
    {
        strncpy(temp.email, new_string, 30);
    }
    else if(strcmp(field, "alias") == 0)
    {
        strncpy(temp.alias, new_string, 30);
    }

    fseek(database, record_number * sizeof(struct _record), SEEK_SET);
    fwrite(&temp, sizeof(struct _record), 1, database);
    fclose(database);
    return true;
}
int main()
{
    char test[160] = "13125657,Shaffer,AUDREY,n/a,n/a,4/4/1933,FEMALE,824-09-8900,87 PARK STREET,n/a,11720,n/a,n/a,"
            "CENTEREACH,NY,814-555-1212,999-999-9999,AVEGA@AMGGT.COM,Scarface";
    loadDatabase(DATABASENAME, test);
    struct _record tmp;
    getRecord(DATABASENAME, 0, &tmp);
    showRecord(&tmp);
    printf("\n");
    printf("\n");
    printf("\n");
    changeRecord(DATABASENAME, 0, "id", "0000001");
    changeRecord(DATABASENAME, 0, "lname", "0000001");
    changeRecord(DATABASENAME, 0, "phone", "0000001");
    getRecord(DATABASENAME, 0, &tmp);
    showRecord(&tmp);
    return 0;
}


