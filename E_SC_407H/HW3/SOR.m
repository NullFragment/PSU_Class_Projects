function [ X, iterations ] = SOR( A, b, x0, relax, max_iter, tol, vars )
%SOR - Uses successive overrelaxation method to determine solution to
% a system of equations.
%   A - input matrix
%   b - right side vector
%   x0 - initial guess
%   relax - relaxation coefficient
%   max_iter - maximum number of iterations before quitting
%   tolerance - error tolerance
%   vars - variables appearing in system of equations

%Check if symbolic vector needs created
if(nargin < 7)
    syms x y z
    vars = [x;y;z];
end

%Create vector of ones to obtain coefficient matrix from variable
% functions.
number_of_ones = size(vars);
one = ones(number_of_ones(1), 1);

%Determine number of iterations
[ m, n ] = size(A);

%%Begin SOR Method
A_plug = subs(A, vars, one);
iterations = 1;
x_curr = x0;
b;
err = 100;
while(iterations < max_iter &&  err > tol)
    x_old = x_curr;
    for i = 1:n
        x_sum = 0;
        for j = 1:n
            if j ~=i
                x_sum = x_sum + A_plug(i,j)*x_curr(j);
            end
        end
        
        x_curr(i) = (1-relax)*x_curr(i) + relax*(b(i) - x_sum)/A_plug(i,i);
    end
    err = norm(max(abs((x_curr - x_old))));
    iterations = iterations + 1;
end

X = x_curr;