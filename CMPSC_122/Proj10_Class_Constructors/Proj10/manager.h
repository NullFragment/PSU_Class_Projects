//File: manager.h
//Manager class definition.



#ifndef _MANAGER_H
#define _MANAGER_H

#include "employee.h"

class Manager : public Employee {
public:
  Manager(string theName,
          float thePayRate,
          bool isSalaried);

  bool getSalaried() const;
  void setSalaried(bool salariedSetting);

  float pay(float hoursWorked) const;

protected:
  bool salaried;
};


//Manager method definitions.
 


Manager::Manager(string theName,
                 float thePayRate,
                 bool isSalaried)
  : Employee(theName, thePayRate)
{
  salaried = isSalaried;
}

bool Manager::getSalaried() const
{
  return salaried;
}

float Manager::pay(float hoursWorked) const
{
  if (salaried)
    return payRate;
  // else 
  return Employee::pay(hoursWorked);
}

void Manager::setSalaried(bool salariedSetting)
{
	salaried = salariedSetting;
}


#endif // not defined _MANAGER_H 

