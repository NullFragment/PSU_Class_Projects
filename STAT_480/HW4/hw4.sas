/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
Sept 14, 2018

This program covers Homework 3 for STAT 480.
********************************************************/

LIBNAME STAT480 'C:\STAT480\';

/*** PROBLEM 1 ***/
DATA STAT480.hw3_p1;
	* Read in using list input;
	INPUT town_name $ town_number month year temp_low_F temp_high_F;
	DATALINES;
	Kane   20  12  2005  12  25
	Ambler 22  12  2005   8  20
	Kane   20  01  2006  13  32
	Oakey  32  12  2005  30  50
	Oakey  32  01  2006  25  45
	Ambler 22  01  2006  15  28
	;
RUN;

PROC PRINT data=STAT480.hw3_p1;
	title 'Output Dataset: STAT480 Homework 3 Problem 1';
RUN;

PROC CONTENTS data=STAT480.hw3_p1;
	/* Format output to hav a max width of 80 chars and
	   be centered */
	OPTIONS LS=80 CENTER;
RUN;

/*** PROBLEM 2 ***/
DATA rats;
	* Read data using formatted input from raw data file;
	INFILE 'C:\STAT480\rats.dat';
	INPUT 
		@1 rat_number 1.
		+2 dob date9.
		@13 disease date9.
		@23 death mmddyy8.
		@32 group $1.
		;
RUN;
PROC PRINT data=rats;
	/* Limit output width to 78, page size to 56 lines,
	   centers the output and suppresses date from printing*/
	OPTIONS LS=78 PS=56 CENTER NODATE;
	title 'Output Dataset: STAT480 Homework 3 Problem 2';
RUN;
