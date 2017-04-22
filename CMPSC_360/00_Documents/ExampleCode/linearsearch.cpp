// Linear search
#include <iostream>
using namespace std;
const int total = 10;
int main()
{	int array[total] = {59, 97, 34, 90, 79, 56, 24, 51, 30, 69};
	int find, location;
	int notFound=1;
	int notExit=1;
	for(int i=0; i<total; i++){
		cout << array[i] << " ";
	}
	do{
		cout<<"\n Please enter an integer value (-1 to quit) : ";
		cin>>find;
		if(find<0){
			notExit = 0;
		}else{
			for(int i=0;i<total;i++){
				if(array[i] == find){
					location=i;
					cout<<" required number is found out at the location: "<<location<<endl;
					notFound=0;
				}
			}
			if(notFound){
				cout<<" Number is not found "<< endl;
			}
			notFound=1;
		}
	}while(notExit);
	cout <<" exit done!"<<endl;
	return 0;
}

