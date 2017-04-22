//CMPSC122 SU Assign3 Phrase 2 Reference Answer 
// File: mystring.cpp
// ==================
// Implementation file for user-defined String class that stores
// characters internally in a static-allocated array.

#include "mystring.h"

String::String()
{
  contents = new char[0];
  len = 0;
}


String::String(const char *s)  // or String::String(const char s[]) 
{
  len = strlen(s);
  contents = new char[len + 1];  // Extra char for nul char (\0). 
  strcpy(contents, s);
}

String::String(const String &str)
{
	len = str.len;
	contents = new(nothrow) char[len];

	if(contents != 0)
	{
		for(int i = 0; i < len; i++)
		{
			contents[i] = str.contents[i];
		}
	}
	else
	{
		cerr << "The given string is empty. Cannot create a copy.";
		exit(1);
	}
}

String & String::operator=(const String & str)
{
   if (this != &str)
   {
	   if(len != str.len)
	   {
		   delete[] contents;
		   len = str.len;
		   contents = new char[len+1];
	   }
	   
	   if(contents == 0)
	   {
			cerr << "The given string is empty. Cannot create a copy.";
			exit(1);
	   }

	   len = str.len;
	   for(int i = 0; i < len; i++)
	   {
		   contents[i] = str.contents[i];
	   }
   }
}


String::~String()
{
	delete [] contents;
}


void String::append(const String &str)
{
  strcat(contents, str.contents);

  len += str.len;

}


bool String::operator ==(const String &str) const
{
  return strcmp(contents, str.contents) == 0;
}

bool String::operator !=(const String &str) const
{
  return strcmp(contents, str.contents) != 0;
}

bool String::operator >(const String &str) const
{
  return strcmp(contents, str.contents) > 0;
}

bool String::operator <(const String &str) const
{
  return strcmp(contents, str.contents) < 0;
}


bool String::operator >=(const String &str) const
{
  return strcmp(contents, str.contents) >= 0;
}


String String::operator +=(const String &str)
{
	append(str);
	return *this;
}

void String::print(ostream &out) const
{
  out << contents;
}

int String::length() const
{
  return len;
}

char String::operator [](int i) const
{
  if (i < 0 || i >= len) {
    cerr << "can't access location " << i
         << " of string \"" << contents << "\"" << endl;
    return '\0';
  }
  return contents[i];
}

ostream & operator<<(ostream &out, const String & s) // overload ostream operator "<<" - External!
{ 
	s.print(out);
	return out;
}