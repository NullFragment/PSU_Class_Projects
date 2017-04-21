// Basic Logic Gate : AND
#include <iostream>
using namespace std;

int main()
{   const int array_size = 2;
    int circuit[array_size] = {1,1};

    if(circuit[0] && circuit[1]){
        cout << "True: ON "<< circuit[0] << " " << circuit[1] << endl;
    }else{
        cout << "False: OFF "<< circuit[0] << " " << circuit[1] << endl;
    }
    return 0;
}
