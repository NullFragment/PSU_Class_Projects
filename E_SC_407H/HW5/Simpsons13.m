function [ Sim13Value ] = Simpsons13( func, x1, x2, intervals )
%SIMPSONS13 Calculates the estimate of an integral using the
% composite Simpson's 1/3 method with an even number of intervals.

% INPUT DEFINITIONS:
% func -> takes in any function with one argument using a function handle
% x1 -> initial value of x
% x2 -> final value of x
% intervals -> number of intervals

% Check if number of intervals is even
    if mod(intervals,2) ~= 0
        error('Number of intervals must be even');
    end

% Initialize step size, inital/final value sum and odd/even sums
    h = (x2-x1)/intervals;
    Sum0 = func(x1) + func(x2);
    Sum1 = 0;
    Sum2 = 0;
% Sum odd and even terms
    for i = 1:intervals-1
        x = x1 + i.*h;
        % Check if odd or even term
        if mod(i,2) == 0
            Sum1 = Sum1+2*func(x);
        else
            Sum2 = Sum2+4*func(x);
        end
    end
    
% Calculate Integral Value
    Sim13Value = (h/3)*(Sum0 + Sum1 + Sum2);
    
end

