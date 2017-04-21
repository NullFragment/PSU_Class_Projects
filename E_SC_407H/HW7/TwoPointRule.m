function [ TwoPtVal ] = TwoPointRule( func, i, j, X, x1, x2 )
%TWOPOINTRULE calculates an integral value using a single application of
% the Two Point Gaussian Quadrature

%INPUT
% func -> Function handle for a function with one input argument
% x1 -> Initial value of x
% x2 -> Final value of x

% Initialize step size, weight, and the two evaluation points
h = (x2-x1)/2;
w = 1/sqrt(3);
xa = (h * -w) + h;
xb = (h*w) + h;

% Calculate Value of the function
TwoPtVal = h* (func(xa,X,i,j) + func(xb,X,i,j));

end