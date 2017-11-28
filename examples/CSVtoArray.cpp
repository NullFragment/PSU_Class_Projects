#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

using namespace std;

int main(){
    int count = 0;
    string dummy;
    int dummy2;
    //int *dummy2;
    //float *arr = (float *)malloc(784*sizeof(float));
    float arr[784];
    for(int i = 0; i < 784; i++)
        arr[i] = 0.0;
    ifstream file("test2CSV.csv");

    string line;
    getline(file, line);

    stringstream iss(line);

    for (int col = 0; col < 784; col++){
        string val;
        getline(iss, val, ',');
        if ( !iss.good() )
            break;

        //printf("%s\t%d\n", val.c_str(), col);


        stringstream convertor(val);
        convertor >> arr[col];
        //val >> arr[col]>> dummy >> dummy2;
        


        //float f = std::stof(val);
        //arr[col] =  f;
    }

    for(int i = 0; i < 784; i++){
        if(arr[i] != 0){
            printf("%f\t%d\n", arr[i], i);
        }
        
        count = i;
    }
    
    //printf("%d\n", count);
    

    return 0;
}