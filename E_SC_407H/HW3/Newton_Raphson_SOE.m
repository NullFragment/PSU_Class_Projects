function [ x, iterations ] = Newton_Raphson_SOE( A, x0, max_iter,tol, vars)
%NEWTON_RAPHSON_SOE  - Uses the Newton-Raphson method to determine 
% solution to a system of equations.
%   A - input matrix
%   b - right side vector
%   x0 - initial guess
%   max_iter - maximum number of iterations before quitting
%   tolerance - error tolerance
%   vars - variables appearing in system of equations

if(nargin < 5)
    syms x y z
    vars = [ x; y; z ];

x_curr = x0;
iterations = 0;
Jac = jacobian(A);
Jac_inverse = inv(Jac);
err = 1;

while(iterations < max_iter && err > tol)
    A_eval = double(subs(A, vars, x_curr));
    J_eval = double(subs(Jac_inverse, vars, x_curr));
    Subtract = J_eval*A_eval;
    x_curr = x_curr - Subtract;
    A_eval = double(subs(A, vars, x_curr));
    err = max(abs(A_eval));
    iterations = iterations+1;
end

x = sym(x_curr);
end
