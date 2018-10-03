/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
Sept 14, 2018

This program covers Homework 5 for STAT 480.
********************************************************/

LIBNAME STAT480 'C:\STAT480\';

/*** PART A ***/
DATA bonescore1;
	* Read data using list input from raw data file;
	INFILE 'C:\STAT480\Bonescor2.dat';
	INPUT singh ccratio csi calcar bone dpa;

	* Calclulate flag 1;
		 IF (singh LE 4)		THEN flag1=1;
	ELSE IF (4 LT singh LE 5)	THEN flag1=2;
	ELSE IF (singh GT 5)		THEN flag1=3;

	* Calclulate flag 2;
		 IF (ccratio GT 0.67)			THEN flag2=1;
	ELSE IF (0.52 LT ccratio LE 0.67)	THEN flag2=2;
	ELSE IF (ccratio LE 0.52)			THEN flag2=3;
	
	* Calclulate flag 3;
		 IF (csi LE 0.55)			THEN flag3=1;
	ELSE IF (0.55 LT csi LE 0.65)	THEN flag3=2;
	ELSE IF (csi GT 0.65)			THEN flag3=3;

	* Calclulate flag 4;
		 IF (calcar LE 6)			THEN flag4=1;
	ELSE IF (6 LT calcar LE 7)		THEN flag4=2;
	ELSE IF (calcar GT 7)			THEN flag4=3;

	* Calclulate ourscore;
	ourscore = flag1 + flag2 + flag3;
RUN;
PROC PRINT data=bonescore1;
	/* Limit output width to 80 and center output */
	OPTIONS LS=80 CENTER;
	title 'Output Dataset: STAT480 Homework 3 Bonescore1 Data';
RUN;
/*** PART B ***/
DATA bonescore2;
	* Read data using list input from raw data file;
	INFILE 'C:\STAT480\Bonescor2.dat';
	INPUT singh ccratio csi calcar bone dpa;

	* Calclulate flag 1;
		 IF (singh LE 4)		THEN flag1=1;
	ELSE IF (4 LT singh LE 5)	THEN flag1=2;
	ELSE IF (singh GT 5)		THEN flag1=3;
	ELSE IF (singh EQ .)		THEN flag1=.;

	* Calclulate flag 2;
		 IF (ccratio GT 0.67)			THEN flag2=1;
	ELSE IF (0.52 LT ccratio LE 0.67)	THEN flag2=2;
	ELSE IF (ccratio LE 0.52)			THEN flag2=3;
	ELSE IF (ccratio EQ .)				THEN flag2=.;

	* Calclulate flag 3;
		 IF (csi LE 0.55)			THEN flag3=1;
	ELSE IF (0.55 LT csi LE 0.65)	THEN flag3=2;
	ELSE IF (csi GT 0.65)			THEN flag3=3;
	ELSE IF (csi EQ .)				THEN flag3=.;

	* Calclulate flag 4;
		 IF (calcar LE 6)			THEN flag4=1;
	ELSE IF (6 LT calcar LE 7)		THEN flag4=2;
	ELSE IF (calcar GT 7)			THEN flag4=3;
	ELSE IF (calcar EQ .)			THEN flag4=.;

	* Calclulate ourscore;
	ourscore = flag1 + flag2 + flag3;
RUN;
PROC PRINT data=bonescore2;
	/* Limit output width to 80 and center output */
	OPTIONS LS=80 CENTER;
	title 'Output Dataset: STAT480 Homework 3 Bonescore2 Data';
RUN;
/*** PART C ***/
DATA bonescore3;
	* Read data using list input from raw data file;
	INFILE 'C:\STAT480\Bonescor2.dat';
	INPUT singh ccratio csi calcar bone dpa;

	* Calclulate flag 1;
	 	 IF (singh EQ .)		THEN flag1=.;
	ELSE IF (singh LE 4)		THEN flag1=1;
	ELSE IF (4 LT singh LE 5)	THEN flag1=2;
	ELSE IF (singh GT 5)		THEN flag1=3;

	* Calclulate flag 2;
		 IF (ccratio EQ .)				THEN flag2=.;
	ELSE IF (ccratio GT 0.67)			THEN flag2=1;
	ELSE IF (0.52 LT ccratio LE 0.67)	THEN flag2=2;
	ELSE IF (ccratio LE 0.52)			THEN flag2=3;

	* Calclulate flag 3;
		 IF (csi EQ .)				THEN flag3=.;
	ELSE IF (csi LE 0.55)			THEN flag3=1;
	ELSE IF (0.55 LT csi LE 0.65)	THEN flag3=2;
	ELSE IF (csi GT 0.65)			THEN flag3=3;

	* Calclulate flag 4;
		 IF (calcar EQ .)			THEN flag4=.;
	ELSE IF (calcar LE 6)			THEN flag4=1;
	ELSE IF (6 LT calcar LE 7)		THEN flag4=2;
	ELSE IF (calcar GT 7)			THEN flag4=3;

	* Calclulate ourscore;
	ourscore = flag1 + flag2 + flag3;
RUN;
PROC PRINT data=bonescore3;
	/* Limit output width to 80 and center output */
	OPTIONS LS=80 CENTER;
	title 'Output Dataset: STAT480 Homework 3 Bonescore3 Data';
RUN;

