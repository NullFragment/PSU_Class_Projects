function[iter, root, root_value, err] = nr_method(x_n, fun, fprime, max_iter, max_error, outputfile)

%Initalize values for tests
iter = 0;
err = 999;
fxn = fun(x_n);
output = fopen(outputfile, 'a');
%Check to see if f' is too close to zero
if(fprime(x_n) < 10^-12)
    error('F-Prime is too close to zero!');
end

%Check to see if given starting point is the root.
if(fxn == 0)
    error('Given x_0 is the root!');
end

%Newton-Raphson Method Iteration
while(iter < max_iter && err > max_error && fxn ~= 0)
    x_n_1 = x_n - (fun(x_n)/fprime(x_n));
    err = abs((x_n_1 - x_n)/(x_n));
    x_n = x_n_1;
    iter = iter+1;
    fxn = fun(x_n);
end

%Set root and root function value.
root = x_n;
root_value = fxn;


%Print all relevant information
fprintf(output, 'Iteration: %s\n', num2str(iter));
fprintf(output, 'Error: %s\n', num2str(err));
fprintf(output, 'Root: %s\n', num2str(root));
fprintf(output, 'Function Value at Root: %s\n', num2str(root_value));
fclose('all');

