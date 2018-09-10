LIBNAME STAT480 'C:\STAT480\';

DATA STAT480.hw3_p1;
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
	OPTIONS LS=80 CENTER;
RUN;
DATA rats;
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
	OPTIONS LS=78 PS=56 CENTER NODATE;
	title 'Output Dataset: STAT480 Homework 3 Problem 2';
RUN;
