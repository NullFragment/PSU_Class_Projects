#include<iostream>
#include<fstream>
#include<string>

using namespace std;

//Data stored in the node type
 
 
struct WordCount
 
{
 
        string word;
 
        int count;
 
};
 
 
 
//Node type: 
 
struct TreeNode
 
{
 
        WordCount info;
 
        TreeNode * left;
 
        TreeNode * right;
 
};
 
 
// Two function's prototype
 
// Increments the frequency count if the string is in the tree
 
// or inserts the string if it is not there.
 
void Insert(TreeNode* node, string str);

 
 
// Prints the words in the tree and their frequency counts.
 
void PrintTree(TreeNode* node , ofstream& file);


void Insert(TreeNode node, string str)
{
	TreeNode currentnode = node, parent;
	bool duplicate = false;

	while(!duplicate && !currentnode.info.word.empty() )
	{
		cout << currentnode.info.word;
		parent = currentnode;
		if(str == currentnode.info.word)
		{
			currentnode.info.count++;
			duplicate = true;
		}
		else if(str < currentnode.info.word)
		{
			currentnode = *currentnode.left;
		}
		else if(str > currentnode.info.word)
		{
			currentnode = *currentnode.right;
		}
	}

	if(!duplicate)
	{
				
		currentnode.info.word = str;
		currentnode.info.count = 1;

		if(parent.info.word.empty())
		{
			node = currentnode;
			
		}

		else if(str < parent.info.word)
		{
			*parent.left = currentnode;
		}
		else if(str > parent.info.word)
		{
			*parent.right = currentnode;
		}
	}


}

void PrintTree(TreeNode node, ofstream &file)
{
	if(!node.info.word.empty())
	{
		PrintTree(*node.left, file);
		file << "Word: " <<node.info.word <<" Count: " <<node.info.count <<endl;
		PrintTree(*node.right, file);
	}
}



//Function Definitions

int main()
{
	string file, word;
	ifstream inputfile;
	ofstream outputfile;
	TreeNode root;

	cout <<"Please enter the name of the file you wish to open: \n";
	
	cin >> file;

	inputfile.open(file);

	while (inputfile >> word)
	{
		Insert(root, word);
	}

	outputfile.open("kps168.txt");

	PrintTree(root, outputfile);
	inputfile.close();
	outputfile.close();

	return 0;
}


