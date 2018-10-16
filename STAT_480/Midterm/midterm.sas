/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
Sept 14, 2018

This program covers the Midterm for STAT 480.
********************************************************/

LIBNAME STAT480 'C:\STAT480\';

/*** QUESTION 1 ***/
DATA question01;
	* Read data using formatted input from raw data file;
	INFILE 'C:\STAT480\question01.dat';	
	INPUT 
		@1	name		$18.
		@20	holes		2.
		@23	par			2. 
		@26	yardage		comma5.
		@32	fees		5.2
	;
RUN;

PROC PRINT data=question01 SPLIT='\' DOUBLE;
	/* 
		Limit output width to 80
		Limit output lines to 58
		Suppress printing date
		Suppress printing output number
		Center output
	*/
	OPTIONS LS=80 PS=58 NODATE NONUMBER CENTER;

	/* Set output variable order */
	var par holes yardage fees;

	/* Set title for observations */
	id name;

	/* Set title */
	title 'Question #1';

	/* Set Labels and output formats */
	label
		name	=	'Golf Course'
		holes	=	'No. of\holes'
		par		=	'Par'
		yardage	=	'Yardage'
		fees	=	'Greens\Fees'
		;

	format 
		yardage comma5.
		fees DOLLAR7.2
		;
RUN;

DATA stat480.question02;
	* Read data using formatted input from raw data file;
	INFILE 'C:\STAT480\question02.dat';	
	INPUT idno name $ team $ strtwght endwght;

	/* Determine weight categories */
		 IF (endwght LT 120)		THEN category = 1;
	ELSE IF (120 LE endwght LT 150)	THEN category = 2;
	ELSE IF (150 LE endwght LT 180)	THEN category = 3;
	ELSE IF (endwght GE 180)		THEN category = 4;
RUN;

PROC PRINT data=stat480.question02 SPLIT='\' DOUBLE;
	/* 
		Limit output width to 80
		Limit output lines to 58
		Suppress printing date
		Suppress printing output number
		Center output
	*/
	OPTIONS LS=80 PS=58 NODATE NONUMBER CENTER;

	/* Set title */
	title 'Question #2';

	/* Set variable output labels */
	label
		idno		=	'ID\Number'
		name		=	'Name'
		team		=	'Team'
		strtwght	=	'Start\Weight'
		endwght		=	'End\Weight'
		category	=	'Category'
		;
RUN;

DATA question03;
	SET 'C:\STAT480\question03.sas7bdat';

	/* Calculate exam averages */
	average1 = (grade1+grade2)/2;
	average2 = (grade2+grade3)/2;

	/* Determine student status values */
		 IF (average1 EQ .) OR (average2 EQ .)	THEN status = "incomplete";
	ELSE IF (average1 EQ average2) 				THEN status = "no change";
	ELSE IF (average1 LT average2) 				THEN status = "improved";
	ELSE IF (average1 GT average2) 				THEN status = "regressed";
RUN;

PROC PRINT data=question03 SPLIT = '\' DOUBLE;
	/* 
		Limit output width to 80
		Limit output lines to 58
		Suppress printing date
		Suppress printing output number
		Center output
	*/
	OPTIONS LS=80 PS=58 NODATE NONUMBER CENTER;

	/* Set title */
	title 'Question #3';

	/* Set variable output order */
	var student grade1 grade2 grade3 average1 average2 status;

	/* Set variable output labels */
	label
		student		= 'Student\Name'
		grade1		= 'Exam 1\Grade'
		grade2		= 'Exam 2\Grade'
		grade3		= 'Exam 3\Grade'
		average1	= 'Exam 1 & 2\Average'
		average2	= 'Exam 2 & 3\Average'
		status		= 'Student\Status'
		;
RUN;
