/* 
// Name: Kyle Salitrik
// ID Number: 997543474
// Project 4
// Due: 7/17/2013, 11:55 PM
// Last Modificatons: 7/17/2013, 6:45
/
// Inputs: Strings
// Outputs: Strings, Intergers
//
// This program evaluates simple postfix expressions using stacks.
// Operators are limited to *, /, +, -.
*/

#include<iostream>
#include<stack>
#include<string>

using namespace std;

int main()
{
	stack<int> operandstack;
	stack<char> operatorstack;
	string input;
	char value;

	cout << "Please enter the postfix operation you would like to evaluate. \nPut a space between each interger and operator. \nEnd your evaluation with a colon ':'";
	cout <<endl;
	getline(cin, input);
	cout << "You entered:\n" <<input <<endl <<endl;
	while ( !input.empty() ) //Begin loop for evaluation
	{
		for(int i=0; input[i] != ':'  && i < input.length() ; i++) //Iteration through string
		{
			value = input[i];
			if( isdigit(value) ) //Conversion from character digit to interger. Pushes value on top of stack.
			{
				int val = value - '0';
				cout << "Token = " <<val <<" Push " <<val <<endl;
				operandstack.push(val);
			}

			else if( isalnum(value) ) //Conversion from character to interger. Pushes value on top of stack.
			{
				int val = value - '0';
				cout << "Token = " <<val <<" Push " <<val <<endl;
				operandstack.push(val);
			}

			else switch(value) //This switch operation determines which operand is to be used and then pushes it on the top of the stack.
			{
			case '+': 
				{
					operatorstack.push(value);
					cout << "Token = " <<value <<" Push " <<value <<endl;
					break;
				}
			case '-':
				{
					operatorstack.push(value);
					cout << "Token = " <<value <<" Push " <<value <<endl;
					break;
				}

			case '*':
				{
					operatorstack.push(value);
					cout << "Token = " <<value <<" Push " <<value <<endl;
					break;
				}
			case '/':
				{
					operatorstack.push(value);
					cout << "Token = " <<value <<" Push " <<value <<endl;
					break;
				}
			case ' ': break;
			}

			if( !operatorstack.empty() ) // Loop for evaluation of expression given.
			{
				int returnval;
				int val1 = operandstack.top(); operandstack.pop(); 
				int val2 = operandstack.top(); operandstack.pop();
				switch ( operatorstack.top() ) // This switching function determines which operation should be taking place from the operand stack.
				{
				case '+':
					{
						returnval = val2 + val1;
						operandstack.push(returnval);
						operatorstack.pop();
						cout << "Token = +" <<" Pop " <<val1 <<" Pop " <<val2 <<" Push " << returnval <<endl;
						break;
					}
				case '-':
					{
						returnval = val2 - val1;
						operandstack.push(returnval);
						operatorstack.pop();
						cout << "Token = -" <<" Pop " <<val1 <<" Pop " <<val2 <<" Push " << returnval <<endl;
						break;
					}
				case '*':
					{
						returnval = val2 * val1;
						operandstack.push(returnval);
						operatorstack.pop();
						cout << "Token = *" <<" Pop " <<val1 <<" Pop " <<val2 <<" Push " << returnval <<endl;
						break;
					}
				case '/':
					{
						returnval = val2 / val1;
						operandstack.push(returnval);
						operatorstack.pop();
						cout << "Token = /" <<" Pop " <<val1 <<" Pop " <<val2 <<" Push " << returnval <<endl;
						break;
					}
				}
			}
		
		}
		int result = operandstack.top(); // Prints final result.
		operandstack.pop();
		cout << "Your result is: " <<result <<endl <<endl;
		cout << "Would you like to run another evaluation? (Y or y for yes)\n"; // Determines if loop should restart.
		char in;
		cin >> in;
		if (in == 'y' || in == 'Y')
		{
			while(!operandstack.empty())
			{
				operandstack.pop();
			}
			while(!operatorstack.empty())
			{
				operatorstack.pop();
			}
			cin.ignore(INT_MAX, '\n');
			cin.clear();
			cout << "\n\nPlease enter new postfix expression.\n";
			getline(cin, input);
			cout << "You entered:\n" <<input <<endl <<endl;
		}
		else break;
	}
	char garbage;
	cout << "Please enter any character to end.\n"; 
	cin >> garbage; // Pauses before closing.
	return 0;
}

/* Execution Trace:

Please enter the postfix operation you would like to evaluate.
Put a space between each interger and operator.
End your evaluation with a colon ':'
5 5 + 3 - 7 / :
You entered:
5 5 + 3 - 7 / :

Token = 5 Push 5
Token = 5 Push 5
Token = + Push +
Token = + Pop 5 Pop 5 Push 10
Token = 3 Push 3
Token = - Push -
Token = - Pop 3 Pop 10 Push 7
Token = 7 Push 7
Token = / Push /
Token = / Pop 7 Pop 7 Push 1
Your result is: 1

Would you like to run another evaluation? (Y or y for yes)
y


Please enter new postfix expression.
1 1 +  2 - 4 + 2 / 3 * :
You entered:
1 1 +  2 - 4 + 2 / 3 * :

Token = 1 Push 1
Token = 1 Push 1
Token = + Push +
Token = + Pop 1 Pop 1 Push 2
Token = 2 Push 2
Token = - Push -
Token = - Pop 2 Pop 2 Push 0
Token = 4 Push 4
Token = + Push +
Token = + Pop 4 Pop 0 Push 4
Token = 2 Push 2
Token = / Push /
Token = / Pop 2 Pop 4 Push 2
Token = 3 Push 3
Token = * Push *
Token = * Pop 3 Pop 2 Push 6
Your result is: 6

Would you like to run another evaluation? (Y or y for yes)
y


Please enter new postfix expression.
A A + 1 - :
You entered:
A A + 1 - :

Token = 17 Push 17
Token = 17 Push 17
Token = + Push +
Token = + Pop 17 Pop 17 Push 34
Token = 1 Push 1
Token = - Push -
Token = - Pop 1 Pop 34 Push 33
Your result is: 33

Would you like to run another evaluation? (Y or y for yes)
n
Please enter any character to end.

*/