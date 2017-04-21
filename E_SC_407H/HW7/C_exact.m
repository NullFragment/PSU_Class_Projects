function [ C ] = C_exact( x )
%C_EXACT Summary of this function goes here
%   Detailed explanation goes here

if x > pi/3
    C = 0.01*sin(3*x);
else
    C = sin(3*x);
end

end

