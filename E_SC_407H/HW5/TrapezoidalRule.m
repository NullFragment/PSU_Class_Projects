function [ TRValue ] = TrapezoidalRule( func, x1, x2, intervals )
%TRAPEZOIDALRULE Calculates the estimate of an integral using the
% composite Trapezoidal Rule.

%INPUT DEFINITIONS:
% func -> takes in any function with one argument using a function handle
% x1 -> initial value of x
% x2 -> final value of x
% intervals -> number of intervals

% Initialize Sum and Step Size
TRSum = 0;
h = (x2-x1)/intervals;

% Calculate Sum
for i = 1:intervals-1
    TRSum = TRSum + func(x1+i*h);
end

% Calculate integral value
TRValue = (h/2)*(func(x1) + 2*TRSum + func(x2));

end

