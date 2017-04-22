
/******************************************************************************
 CMPSC122 Assignment: Project #8 -- sample driver
 ******************************************************************************/

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "timer.h"
using namespace std;

int Max_Subsequence_Sum_BLUE( const int A[], const int N )
{
	int This_Sum = 0, Max_Sum = 0;
	
	for (int i=0; i<N; i++)
	{
		This_Sum = 0;
		for (int j=i; j<N; j++)
		{
			This_Sum += A[j];
			if (This_Sum > Max_Sum)
			{
				Max_Sum = This_Sum;
			}
		}
	}
	return Max_Sum;
}

int Max_Subsequence_Sum_GREEN( const int A[], const int N )
{
  int This_Sum = 0, Max_Sum = 0;

  for (int i=0; i<N; i++)
  {
    for (int j=i; j<N; j++)
    {
      This_Sum = 0;
      for (int k=i; k<=j; k++)
      {
        This_Sum += A[k];
      }
      if (This_Sum > Max_Sum)
      {
        Max_Sum = This_Sum;
      }
    }
  }
  return Max_Sum;
}

int Max_Subsequence_Sum_RED( const int A[], const int N )
{
  int This_Sum = 0, Max_Sum = 0;

  for (int Seq_End=0; Seq_End<N; Seq_End++)
  {
    This_Sum += A[Seq_End];

    if (This_Sum > Max_Sum)
    {
      Max_Sum = This_Sum;
    }
    else if (This_Sum < 0)
    {
      This_Sum = 0;
    }
  }
  return Max_Sum;
}

int main( )
{
	int Size = 64;
	int *Vec, Result[6];
	char Answer;
	Timer T[6];
	
	for ( int i = 0; i < 6; i++)
	{ 
		Vec = new int [Size];
		srand( time(0) );
		
		for (int I=0; I<Size; I++)
		{
			Vec[I] = rand() % 100 - 50;
		}
		
		cout << "Do you wish to view the array contents? (Y or N): ";
		cin >> Answer;
		
		if (Answer == 'Y' || Answer == 'y')
		{
			for (int J=0; J<Size; J++)
			{
				cout << Vec[J] << "\n";
			}
		}
		cout << endl;
		
		T[i].start();
		Result[i]  = Max_Subsequence_Sum_RED( Vec, Size );
		T[i].stop();

		Size = 2* Size;
	}
	
	for ( int i = 0; i < 6; i++)
	{ 
		cout << "Maximum contiguous subsequence sum: " << Result [i] << "\n";
		T[i].show();
		cout << endl;
	}
	char garbage;
	cin >>garbage;
}