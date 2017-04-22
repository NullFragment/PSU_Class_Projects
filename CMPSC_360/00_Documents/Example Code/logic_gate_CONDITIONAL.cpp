// Basic Logic Gate : CONDITIONAL
#include <iostream>
using namespace std;
int main()
{   int logic[2];
    const int array_size = 2;
    int circuit[array_size] = {1,0};
    // boating Sunday if not raining
    if(circuit[0] && circuit[1]){
        cout << "True: ON - boating "<< circuit[0] << " " << circuit[1] << endl;
    }else{
        if(!(circuit[0] && !circuit[1])){
            cout << "True: ON - boating "<< circuit[0] << " " << circuit[1] << endl;
        }else{
            cout << "False: OFF - not boating "<< circuit[0] << " " << circuit[1] << endl;
        }
    }
    return 0;
}