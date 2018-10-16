/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
Sept 14, 2018

This program covers Homework 6 for STAT 480.
********************************************************/

LIBNAME STAT480 'C:\STAT480\';

/*** PART A ***/
DATA books;
	* Read data using formatted input from raw data file;
	INFILE 'C:\STAT480\kidsbooks.txt';	
	INPUT 
		@1	Title		$20.
		@22	Author		$19.
		@43	Publisher	$16. 
		@61	Pubdate		mmddyy10.
		@72	Price		5.2
		@79 Pages		2.
	;

	/* Calculate price per page for each book */
	PricePerPage=Price/Pages;
RUN;

PROC SORT data=books out=sortedBooks;
	by descending price;
RUN;

PROC PRINT data=sortedBooks SPLIT='/' DOUBLE;
	/* Limit output width to 80 and center output */
	OPTIONS LS=80 PS=38 NODATE NONUMBER;

	/* Set title for observations */
	id Title;

	/* Select values to print out */
	var Author Price Pages PricePerPage Publisher Pubdate;
	where pricePerPage > 0.30;
	sum Price;

	/* Set Labels and output formats */
	label
		Title = 'Title'
		Author = 'Author'
		Price = 'Price'
		Pages = 'No. of/Pages'
		PricePerPage = 'Price/Per Page'
		Publisher = 'Publisher'
		Pubdate = 'Publication/Date';
	format 
		Pubdate DATE9.
		Price DOLLAR6.2
		PricePerPage DOLLAR5.2;

	/* Set title and footnote */
	title 'Popular Books for Children';
	footnote 'Price obtained from Amazon.com';
RUN;
