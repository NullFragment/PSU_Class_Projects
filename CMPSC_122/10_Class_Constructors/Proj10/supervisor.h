/*
* file: supervisor.h
* supervisor class definition
*/

#ifndef _SUPERVISOR_H
#define _SUPERVISOR_H

#include "employee.h"

class Supervisor : public Manager
{
public:
	Supervisor(string theName, 
		float thePayRate, 
		string theDepartment);

	string getDept();
	void setDepartment(string newDepartment);

	float pay(float value);

protected:
	string department;
};

//Supervisor definitions

Supervisor::Supervisor(string theName, 
		float thePayRate, 
		string theDepartment)
		:Manager(theName, thePayRate, true)
{
	department = theDepartment;
}

string Supervisor::getDept()
{
	return department;
}

void Supervisor::setDepartment(string newDepartment)
{
	department = newDepartment;
}

float Supervisor::pay(float value)
{
	return payRate;
}

#endif