/********************************************************
Kyle Salitrik
kps168
PSU ID: 997543474
Oct 28, 2018

This program covers Homework 8 for STAT 480.
********************************************************/

DATA sales (DROP = Mon Tues Wed Thur Fri);
    INPUT 
		@1 weekof ddmmyy8. 
		@10 Store 3.
		@14 Mon 3.
		@18 Tues 3. 
		@22 Wed 3.
		@26 Thur 3.
		@30 Fri 3.; 

	* Calculate average sales;
	AvgSales = (Mon + Tues + Wed + Thur + Fri)/5; 	
	
	* Set length of new group variable;
	length Group $7;

	* Determine whether sales are low, avg, high, or missing.;
	if AvgSales = '.' then Group = 'N/A'; 
	else if AvgSales LE 605 then Group = 'Low';
	else if 605 LT AvgSales LE 750 then Group = 'Average'; 
	else if AvgSales GT 750 then Group = 'High'; 

	* Set region based on store number;
	if (110 LE Store LE 111) then region = 'South'; 
	else if 112 LE Store LE 114 then region = 'North';

	DATALINES;
10/12/07 110 412 532 641 701 802
10/12/07 111 478 567 699 789 821
10/12/07 112 399 501 650 712 812
10/12/07 113 421 532 698 756 872
10/12/07 114 401 510 612 721 899
17/12/07 110 710 725 789 721 799
17/12/07 111 689 701 729 703 721
17/12/07 112 899 812 802 738 712
17/12/07 113 700 712 748 765 801
17/12/07 114 699 799 899 608 .	
24/12/07 110 340 333 321 401 490
24/12/07 111 801 793 721 763 798
24/12/07 112 598 798 684 502 412
24/12/07 113 980 921 832 812 849
24/12/07 114 798 709 721 799 724
31/12/07 110 487 321 399 312 321
31/12/07 111 501 532 598 581 601
31/12/07 112 598 512 540 523 549
31/12/07 113 601 625 674 698 601
31/12/07 114 900 805 700 601 811
	;
RUN; 

PROC PRINT data = sales label;
	* Set page size to 50, line size to 80, and suppress date and number output;
	OPTIONS PS = 58 LS = 80 NODATE NONUMBER;
	
	* Set output title;
	title "Sales Data";
	 
	*Set label and format of week output.;
	label weekof = 'Date';
	format weekof ddmmyy8.;
RUN;
