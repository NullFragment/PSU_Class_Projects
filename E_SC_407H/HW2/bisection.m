function[iter, root, root_value, err] = bisection(a, b, fun, max_iter, max_error, outputfile)

%Initalize values for tests
iter = 0;
err = 999;
root = 999;
output = fopen(outputfile, 'a');
%Test to see if a nd b have the same values or if a > b
% If a > b, swap a and b
if( a == b )
    fprintf('Please specify valid endpoints.');
    error('Endpoints are equal. Cannot evaluate.');
elseif( a > b )
    c = b;
    a = b;
    b = c;
    clear c;
end

%Test to see if interval is valid.
fa = fun(a);
fb = fun(b);
test = fa * fb;
if ( test >= 0 )
    fprintf('Given interval is not suitable for calculating the root.');
    error('Value of fa * fb is greater than or equal to zero.');
end


%Run first iteration of the bisection method manually for while loop
%to work
c_old = (a+b)/2;
fc = fun(c_old);
if(fc < fb)
    b = c_old;
elseif (fc > fa)
    a = c_old;
end

%Iteration of bisection method
while(iter < max_iter && err > max_error && fc ~= 0)
    iter = iter + 1;
    c_curr = (a+b)/2;
    err = abs((c_curr - c_old)/(c_curr));
    fa = fun(a);
    fb = fun(b);
    fc = fun(c_curr);
    %Test to determine which interval value should move towards the root
    if(fc > 0)
        b = c_curr;
    elseif (fc < 0)
        a = c_curr;
    end
    c_old = c_curr;
end

%Set values for output
root = c_old;
root_value = fc;

%Print all relevant information
fprintf(output, 'Iteration: %s\n', num2str(iter));
fprintf(output, 'Error: %s\n', num2str(err));
fprintf(output, 'Root: %s\n', num2str(root));
fprintf(output, 'Function Value at Root: %s\n', num2str(root_value));
fclose('all');



    