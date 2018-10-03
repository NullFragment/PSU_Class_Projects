/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
Sept 23, 2018

This program covers Homework 4 for STAT 480.
********************************************************/

/*** PROBLEM 1 ***/
DATA dietdata;
	* Read data using column input from raw data file;
	INFILE 'C:\STAT480\dietdata.dat';
	INPUT subj $ 1-3 height $ 4-5 wt_init 6-8 wt_final 9-11;
	height_m = input(height, 2.)*0.0254; * Casts height to numeric value and converts to meters;
	bmi_init = (wt_init/2.2)/(height_m**2); * Calculates initial BMI;
	bmi_final = (wt_final/2.2)/(height_m**2); * Calculates final BMI;
	bmi_diff = bmi_final - bmi_init; * Calculate difference of final minus initial BMI;
RUN;

PROC PRINT data=dietdata;
	* Set line size to 80 and page size to 58 and print data;
	OPTIONS LS=80 PS=58; 
	title 'Output Dataset: STAT480 Homework 4 Diet Data';
RUN;

/*** PROBLEM 2 ***/
DATA temp;
	* Input temp data and run arithmetic calculations;
	* Three was ambiguous saying "difference between" so I performed both calculations ;
	INPUT abc def ghi jkl;
	one = abc + def - ghi + jkl; /* 10+5-2+4 = 17*/
	two = (abc + def) - (ghi + jkl); /*(10+5)-(2+4) = 15-6 = 9 */
	three = abc + jkl + (def - ghi); /* 10+4+(5-2) = 14+3 = 17 */
	three_2 = abc + jkl + (ghi - def); /* 10+4+(2-5) = 14-3 = 11 */
	four = abc + jkl + def / ghi; /* 10+4+5/2 = 14+2.5 = 16.5 */
	five = (abc + def)/(ghi + jkl); /*(10+5)/(2+4) = 15/6 = 2.5*/
	DATALINES;
		10 5 2 4
	;
RUN;

PROC PRINT data=temp;
	* Set line size to 80 and page size to 58 and print data;
	OPTIONS LS=80 PS=58;
	title 'Output Dataset: STAT480 Homework 4 Arithmetic Data';
RUN;

