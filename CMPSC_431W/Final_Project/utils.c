#include "utils.h"

// #############################################################################
// ### UTILITY FUNCTIONS
// #############################################################################

/**
 * @brief Trims whitespace from a given character array
 * @param to_trim - pointer to array to trim whitespace from
 */
void trimwhitespace(char *to_trim)
{
    char *j;
    while (isspace(*to_trim))
    {
        to_trim++;
    }
    size_t length = strlen(to_trim);
    j = to_trim + length - 1;
    while (isspace(*j))
    {
        *j = 0;
        j--;
    }
}

/**
 * @brief Trims quotes from a given character array
 * @param string - pointer to array to trim whitespace from
 */
char *trimChars(char *string, char *to_trim)
{
    char *j;
    while (strncmp(string, to_trim, 1) == 0)
    {
        string++;
    }
    size_t length = strlen(string);
    j = string + length - 1;
    while (strcmp(j, to_trim) == 0)
    {
        *j = 0;
        j--;
    }
    return string;
}

bool compareStrings(char *string1, char *string2, size_t length, int comparison)
{
    if (length != 0)
    {
        switch (comparison)
        {
            case 0:
            {
                if (strncmp(string1, string2, length) == 0) return true;
                else return false;
            }
            case -1:
            {
                if (strncmp(string1, string2, length) < 0) return true;
                else return false;
            }
            case -2:
            {
                if (strncmp(string1, string2, length) <= 0)
                    return true;
                else return false;
            }
            case 1:
            {
                if (strncmp(string1, string2, length) > 0) return true;
                else return false;
            }
            case 2:
            {
                if (strncmp(string1, string2, length) >= 0) return true;
                else return false;
            }
            default:
            {
                fprintf(stderr, "Invalid case entered -- returning failure");
                return false;
            }
        }
    }
    else
    {
        switch (comparison)
        {
            case 0:
            {
//                int cmpVal = strcmp(string1, string2);
//                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) == 0) return true;
                else return false;
            }
            case -1:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) < 0) return true;
                else return false;
            }
            case -2:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) <= 0) return true;
                else return false;
            }
            case 1:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) > 0) return true;
                else return false;
            }
            case 2:
            {
                int cmpVal = strcmp(string1, string2);
                printf("Compare 1: %s, Compare 2: %s, CmpVal: %d, Case: %d\n", string1, string2, cmpVal, comparison);
                if (strcmp(string1, string2) >= 0) return true;
                else return false;
            }
            default:
            {
                fprintf(stderr, "Invalid case entered -- returning failure");
                return false;
            }
        }
    }
}