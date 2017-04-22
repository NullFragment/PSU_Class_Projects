// Binary Relation : Relation Matrix
#include <iostream>
using namespace std;
int main()
{   int i,k,p,winner_points,winner_id;
    const int total_A = 4;
    const int total_B = 4;
    const int total_C = 2;
    struct relation_matrix{
        string name;
        int points;
        int B[total_B];
        int C[total_C];
    }A[total_A] = { {"Mary",  0, 1, 1, 1, 0, 1, 1},
                    {"David", 0, 0, 1, 1, 1, 0, 1},
                    {"Karen", 0, 1, 1, 1, 1, 1, 1},
                    {"Paul",  0, 1, 1, 0, 1, 1, 1}
                  };
    for(i=0;i<total_A;i++){
           for(k=0;k<total_B;k++){
                A[i].points+=A[i].B[k];
           }
           for(p=0;p<total_C;p++){
                A[i].points+=A[i].C[p];
           }
    }
    winner_points = 0;
    for(i=0;i<total_A;i++){
    cout << "name : " << A[i].name << " - Points : " << A[i].points<< endl;
        if(winner_points<A[i].points){
            winner_points=A[i].points;
            winner_id= i;
        }
    }
    cout << "Who was hired : "
         <<  A[winner_id].name << " - Points : " << A[winner_id].points<< endl;
    return 0;
}

