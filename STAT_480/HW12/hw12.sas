/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
December 2, 2018

This program covers Homework 12 for STAT 480.
********************************************************/

LIBNAME STAT480 'C:\STAT480\';

* Create formats for data;
PROC FORMAT;
	* Create value format for country;
	VALUE sexFmt 1 = 'Male'
				 2 = 'Female';

	* Create a value format for marital status;
	VALUE marStFmt   1 = 'Married'
				     2 = 'Partner'
				     3 = 'Separated'
				     4 = 'Divorced'
				     5 = 'Widowed'
				     6 = 'Never';
RUN;

DATA icdbTemp;
	* Load in background dataset to a temporary data set;
	SET STAT480.back;
RUN;

* Problem 1;
PROC FREQ data=icdbTemp;
	OPTIONS LS = 80 NODATE NONUMBER;
	tables ed_level/nocum;
RUN;	

* Problem 2;
PROC FREQ data=icdbTemp;
	OPTIONS LS = 80 NODATE NONUMBER;
	tables sex*ed_level/nocum nocol nopercent;
RUN; 

* Problem 3;
PROC FREQ data=icdbTemp;
	OPTIONS LS = 80 NODATE NONUMBER;
	tables sex*mar_st/out=summary nocum nocol nopercent noprint sparse;
RUN; 

PROC PRINT;
	FORMAT 
		sex sexFmt.
		mar_st marStFmt.;
RUN;
