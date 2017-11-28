#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* calloc, exit, free */

void ReadCSV(std::ifstream &file, int elements, float *array)
{
    std::string csvData;
    getline(file, csvData);

    std::istringstream dataStream(csvData);

    for (int col = 0; col < elements; col++){
        std::string value;
        getline(dataStream, value, ',');
        if ( !dataStream.good() )
            break;
        std::istringstream convertor(value);
        convertor >> array[col];
    }
}

int main(){
    float *arr;
    arr = (float *)calloc(784, sizeof(float));
    std::ifstream file("train_img.csv");

    ReadCSV(file, 784, arr);
    for(int i = 0; i < 784; i++)
    {
        if(arr[i] != 0) printf("%f\t%d\n", arr[i], i);
    }

    ReadCSV(file, 784, arr);
    for(int i = 0; i < 784; i++)
    {
        if(arr[i] != 0) printf("%f\t%d\n", arr[i], i);
    }
    return 0;
}