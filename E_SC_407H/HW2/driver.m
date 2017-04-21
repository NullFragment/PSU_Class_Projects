function driver()
clear
clc

%Open file for output
outputfile = 'output.txt';
output = fopen(outputfile, 'w');

%Run the xn functon twice to save an easier to see resolution graph
%for the smaller numbers.
fprintf('------------------------PROBLEM #1------------------------\n');
fprintf('\n------------------------RUN N = 30------------------------\n');
xn(30, 'xn30');
fprintf('\n----------------------RUN N = 37------------------------\n');
xn(37, 'xn37');


%Begin printing to output file for Bisection, Newton and Secant Methods
fprintf(output,'------------------------PROBLEM #2------------------------');
fprintf(output,'\n---------------------BISECTION METHOD---------------------\n\n');
fclose('all');

%Close file before bisection method uses it, time bisection method and
%print it out along with separation for Newton's method.
tic;
bisection(300, 500, @fun, 200, 10^-16, outputfile);
bisection_time = toc;

output = fopen(outputfile, 'a');
fprintf(output,'Bisection Method Time: %s\n', num2str(bisection_time));
fprintf(output,'\n\n----------------------NEWTONS METHOD----------------------\n\n');
fclose('all');

%Close file before Newton's method uses it, time Newton's method and
%print it out along with separation for Secant method.
tic;
nr_method(300, @fun, @fprime, 200, 10^-16, outputfile);
nr_time = toc;

output = fopen(outputfile, 'a');
fprintf(output,'Newton Method Time: %s\n', num2str(nr_time));
fprintf(output,'\n\n------------------------SEC METHOD------------------------\n\n');
fclose('all');

%Close file before secant method uses it, time secant method and
%print it out.
tic;
secant_method(400, @fun, 200, 10^-16, outputfile);
sec_time = toc;

output = fopen(outputfile, 'a');
fprintf(output,'Secant Method Time: %s\n', num2str(nr_time));
fprintf(output,'\n\n------------------------ F-ZERO ------------------------\n\n');
fclose('all');

tic;
fzero(@fun, 9);
fzero_time = toc;

output = fopen(outputfile, 'a');
fprintf(output,'F-Zero Method Time: %s\n', num2str(nr_time));
fclose('all');


%Print information regarding the finished program.

fprintf('\n\n***************************************************************************\n\n')
fprintf('Problem 1 is printed on the screen above. And graphs have been\n')
fprintf('made as jpgs in the folder this was run from for 30 and 37\n')
fprintf('iterations. The output for problem 2 has been put into a file\n')
fprintf('output.txt in the folder this was run as well.')
fprintf('\n\n***************************************************************************\n\n')
end
    