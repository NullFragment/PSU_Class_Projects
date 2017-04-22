/*
 * File: employee.h
 * Employee class definition.
 */

#ifndef _EMPLOYEE_H
#define _EMPLOYEE_H

#include <string>
using namespace std;

class Employee {
public:
  Employee(string theName, float thePayRate);
  string getName() const;
  float getPayRate() const;
  float pay(float hoursWorked) const;
  void setName(string newName);
  void setPayRate(float newPayRate);

protected:
  string name;
  float payRate;
};

/*
 * Employee method definitions.
 */

Employee::Employee(string theName, float thePayRate)
{
  name = theName;
  payRate = thePayRate;
}

string Employee::getName() const
{
  return name;
}

float Employee::getPayRate() const
{
  return payRate;
}

float Employee::pay(float hoursWorked) const
{
  return hoursWorked * payRate;
}

void Employee::setName(string newName)
{
	name = newName;
}

void Employee::setPayRate(float newPayRate)
{
	payRate = newPayRate;
}


#endif /* not defined _EMPLOYEE_H */

