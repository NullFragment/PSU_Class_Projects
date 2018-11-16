/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
November 4, 2018

This program covers Homework 9 for STAT 480.
********************************************************/

LIBNAME STAT480 'C:\STAT480\';
DATA states;
	set STAT480.state_cd (rename = (code = start name = label));
	fmtname = 'state2fmt';
RUN;

* Create formats for data;
PROC FORMAT cntlin=states;
	* Create picture for Subject;
	PICTURE subjPic LOW-HIGH = '00-00000' (PREFIX='#');

	* Create a picture for r_id;
	PICTURE ridPic LOW-HIGH = '9999';

	* Create value format for country;
	VALUE countryFmt 1 = 'United States'
				     2 = 'Canada'
				     3 = 'Mexico'
				     OTHER = 'Other';

	* Create value format for race;
	VALUE raceFmt    3 = 'White'
				     4 = 'Black'
				     OTHER = 'Other';

	* Create a value format for marital status;
	VALUE marStFmt   1 = 'Married'
				     2 = 'Living with a partner'
				     3 = 'Separated'
				     4 = 'Divorced'
				     5 = 'Widowed'
				     6 = 'Never married';
RUN;

DATA backTemp;
	* Load in dataset;
	SET STAT480.back;
	KEEP subj r_id country race state mar_st;
RUN;

PROC PRINT DATA=backTemp(OBS=10);
	OPTIONS LS=80 NODATE NONUMBER;
	title 'Demonstration of Subject Picture';
	VAR subj;
	FORMAT  subj subjPic.;
RUN;

PROC FREQ data=backTemp;
	title 'Frequency Count of All other Variables';
	format
		r_id	ridPic.
		country	countryFmt.
		race	raceFmt.
		state	state2fmt.
		mar_st	marStFmt.;
	table r_id country race state mar_st;
RUN;

 PROC FORMAT FMTLIB;
 	title 'Format Library';
 RUN;
