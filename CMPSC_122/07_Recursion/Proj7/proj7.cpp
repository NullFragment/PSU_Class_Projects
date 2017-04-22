/**
//
// Name: Kyle Salitrik
// ID: 997543474
//
// CMPSC 122 Summer 2013
//
// Due Date: 11:55PM 7/24/2013
// Last Modifications: 8:40 7/24/2013
//
// Description: Finds the number of paths between two points traveling northeast.
//
// Input: Intergers.
// Output: Strings, Intergers
//
**/


#include<iostream>
#include"timer.h"

using namespace std;

int pathcount = 0;

int RecursivePath(int endx, int endy, int currentx, int currenty) //Finds the number of NE-paths between two points recursively.
{
	if( currentx < endx+1 && currenty < endy + 1)
	{
		RecursivePath (endx, endy, currentx + 1, currenty);
		RecursivePath (endx, endy, currentx, currenty+1);
		if(currentx == endx && currenty == endy)
		{
			return pathcount++; // Increment path count for return variable.
		}
	}
}


int main()
{
	int x,
	y;
	char answer = 'y';

	while(answer == 'y' || answer == 'Y')
	{
		cout << "Please enter how many units north point B is: \n";
		cin >> y;
		cout <<"\nPlease enter how many units east point B is: \n";
		cin >> x;

		Timer a("Path Clock");

		a.start();
		RecursivePath(x, y, 0, 0);
		a.stop();

		cout<<endl <<endl <<"There are " <<pathcount <<" northeast paths between points A and B.\n\n";
		pathcount = 0; //reset pathcount variable
		a.show();
		cout<< "\n\nWould you like to test another path? (Y or y for yes)\n";
		cin >> answer;
	}
}

/***

Execution:

Please enter how many units north point B is:
2

Please enter how many units east point B is:
3


There are 10 northeast paths between points A and B.

  Path Clock
  -------------------------------
  Elapsed Time   : 0.001s


Would you like to test another path? (Y or y for yes)
y
Please enter how many units north point B is:
5

Please enter how many units east point B is:
7


There are 792 northeast paths between points A and B.

  Path Clock
  -------------------------------
  Elapsed Time   : 0.001s


Would you like to test another path? (Y or y for yes)
y
Please enter how many units north point B is:
12

Please enter how many units east point B is:
14


There are 9657700 northeast paths between points A and B.

  Path Clock
  -------------------------------
  Elapsed Time   : 1.87s


Would you like to test another path? (Y or y for yes)
n

***/
