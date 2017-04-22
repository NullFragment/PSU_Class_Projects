# ifndef _MYPERSON_H
# define _MYPERSON_H

#include <iostream>
#include <cstring>

using namespace std;

class Person
{
public:
	Person(); // Default Constructor
	Person(char name[], int id); // Initialized constructor
	Person(const Person &person); // Copy constructor
	Person & operator = (const Person &person);
	~Person(); // Destructor
	int get_id() const; // Get ID number from object.
	char get_name() const;
	void set_name(char * new_name[]); // Set a new name

private:
	int len;
	char * current_name;
	int current_id;
};

#endif

ostream & operator<<(ostream &out, const Person &person);

/* Begin Class Method Definitions */

Person::Person()
{
	len = strlen("Anonymous");
	current_name = new char[len+1];
	strcpy(current_name, "Anonymous");
	current_id = 123456789;
}

Person::Person(char name[], int id)
{
	len = strlen(name);
	current_name = new char[len+1];
	strcpy(current_name, name);
	current_id = id;
}

Person & Person::operator=(const Person &person)
{
	if( this != &person)
	{
		current_id = person.get_id;
		if(len < person.len)
		{
			delete [] current_name;
			len = person.len;
			current_name = new char[len+1];
		}

		if(person.current_name == 0)
		{
			cerr << "Cannot create copy of empty name.";
			exit(1);
		}

		for(int i = 0; i <= len; i++)
		{
			current_name[i] = person.current_name[i];
		}
	}

}

Person::~Person()
{
	delete [] current_name;
}



int Person::get_id() const
{
	return current_id;
}

char Person::get_name() const
{
	return *current_name;
}


void Person::set_name(char * new_name[])
{
	if(len < strlen(*new_name))
	{
		delete [] current_name;
		current_name = new char [strlen(*new_name) + 1];
	}

	strcpy(current_name, *new_name);
}

/* End Class Method Definitions */

ostream & operator<<(ostream &out, const Person &person)
{
	out << "Person's Name: " << person.get_name << " (" <<person.get_id << ")";
	return out;
}
