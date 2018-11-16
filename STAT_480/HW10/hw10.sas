/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
November 11, 2018

This program covers Homework 10 for STAT 480.
********************************************************/

LIBNAME STAT480 'C:\STAT480\';

* Create formats for data;
PROC FORMAT;
	* Create value format for country;
	VALUE sexFmt 1 = 'Male'
				 2 = 'Female';
	VALUE incomeFmt 1 = 'LT $30,000'
					2 = 'GT $30,000';
RUN;

DATA painTemp;
	* Load in dataset for problem 1;
	SET STAT480.analysis;
	KEEP subj sex income purg_1;
RUN;

PROC REPORT DATA=painTemp NOWINDOWS HEADLINE;
	* Suppress date and output number;
	OPTIONS NODATE NONUMBER;

	* Create report for problem 1;
	title 'ICDB Study Pain Score';

	* Select columns to print out;
	column sex income purg_1;

	* Define sex output and group by gender;
	define sex / group 'Gender' format=sexFmt.;

	* Define income formatting and output;
	define income / across 'Household Income' format=incomeFmt.;

	* Calculate average and define output format for pain scores;
	define purg_1 / mean format = 5.4 'Pain/Score';
RUN;

DATA parkTemp;
	* Load in dataset for problem 2;
	INFILE 'C:\STAT480\natparks.dat';
	INPUT parkName $ 1-21 type $ region $ museums camping;
	
RUN;

PROC REPORT DATA=parkTemp NOWINDOWS HEADSKIP;
	* Print report for problem 2;
	title 'National Parks';

	* Set wanted columns;
	column region museums camping facilities;
	define Region / group 'Region';

	* Set format for museums and camping to center properly;
	define Museums / 'Museums' format = 2.0 center width = 7;
	define Camping / 'Camping' format = 2.0 center width = 7;

	* Define facilities computation and format;
	define facilities / computed 'Facilities' format = 2.0 center width = 10;
	compute facilities;
		facilities = museums.sum + camping.sum;
	endcomp; 
RUN;
