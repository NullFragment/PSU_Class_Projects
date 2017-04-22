// Basic Logic Gate : Truth Table | CONDITIONAL
#include <iostream>
using namespace std;
int main()
{   int i;
    const int total_columns = 2;
    const int total_rows = 4;
    struct mydata{
        int gate[total_columns];
    }circuit[total_rows] = { { 1, 1 },
                             { 1, 0 },
                             { 0, 1 },
                             { 0, 0 } } ;
    cout << "                          Truth Table" << endl;
    for(i=0;i<4;i++){
        // boating Sunday if not raining
        if(circuit[i].gate[0] && circuit[i].gate[1]){
            cout << "True  : boating     - ON:  "<< circuit[i].gate[0]
                    << " " << circuit[i].gate[1] << " | 1" << endl;
        }else{
            if(!(circuit[i].gate[0] && !circuit[i].gate[1])){
                cout << "True  : boating     - ON:  "<< circuit[i].gate[0]
                    << " " << circuit[i].gate[1] << " | 1" << endl;
            }else{
                cout << "False : not boating - OFF: "<< circuit[i].gate[0]
                    << " " << circuit[i].gate[1] << " | 0" << endl;
            }
        }
    }
    return 0;
}