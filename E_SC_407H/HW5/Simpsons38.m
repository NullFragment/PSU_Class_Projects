function [ Sim38Val ] = Simpsons38( func, x1, x2, intervals )
%SIMPSONS38 Calculates the estimate of an integral using the
% composite Simpson's 3/8 method with a mumber of intervals divisible by 3.

%INPUT DEFINITIONS:
% func -> takes in any function with one argument using a function handle
% x1 -> initial value of x
% x2 -> final value of x
% intervals -> number of intervals

% Check if number of intervals is divisible by 3
if mod(intervals,3) ~= 0;
    error('Number of intervals must be divisible by 3!');
end

% Initialize step size, inital/final value sum and 3*/2* sums
h = (x2-x1)/intervals;
Sum1 = 0;
Sum2 = 0;
Sum = func(x1) + func(x2);

% Calculate sum of values multiplied by 3
% n = 2,3,5,6....
for i = 2:3:intervals-1
    xa = x1 + i*h;
    xb = x1 + (i+1) * h;
    Sum1 = Sum1 + 3*(func(xa) + func(xb));
end

% Calculate sum of values multiplied by 2
% n = 4,7,10...
for i = 4:3:intervals-2
    x = x1 + i*h;
    Sum2 = Sum2 + 2*func(x);
end

% Calculate integral value
Sim38Val = (3*h/8)*(Sum + Sum1 + Sum2);

end

