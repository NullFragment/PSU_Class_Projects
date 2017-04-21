function[iter, root, root_value, err] = secant_method(x_nm_1, fun, max_iter, max_error, outputfile)

%Initalize values for tests
iter = 0;
err = 999;
fxn = fun(x_nm_1);
x_n = x_nm_1 + 10^-8;
output = fopen(outputfile, 'a');

%Check to see if given starting point is the root.
if(fxn == 0)
    error('Given x_0 is the root!');
end

%Secant Method Iteration
while(iter < max_iter && err > max_error && fxn ~= 0)
    %Using x_n+1 = x_n - f(x_n)/Q(x_n-1, x_n),
    %where Q = [f(x_n-1) - f(x_n)]/[x_n-1 - x_n]:
    %diff 1 and 2 are the numerator and denominator of Q, respectively,
    %and denominator is the value of Q.
    diff1 = fun(x_nm_1) - fun(x_n);
    diff2 = (x_nm_1 - x_n);
    denominator = diff1/diff2;
    
    x_np_1 = x_n - (fun(x_n) / denominator);
    
    fxn = fun(x_np_1);
    err = abs(fxn);
    
    x_nm_1 = x_n;
    x_n = x_np_1;
    
    iter = iter + 1;
end

%Set root and root function value.
root = x_n;
root_value = fun(x_n);

%Print all relevant information
fprintf(output, 'Iteration: %s\n', num2str(iter));
fprintf(output, 'Error: %s\n', num2str(err));
fprintf(output, 'Root: %s\n', num2str(root));
fprintf(output, 'Function Value at Root: %s\n', num2str(root_value));
fclose('all');

