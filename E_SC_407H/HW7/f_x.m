function [ F ] = f_x( x, X, i, j )
%F_X Summary of this function goes here
%   Detailed explanation goes here
if x > (pi/3)
    F = 9.01*sin(3*x);
else
    F = 10*sin(3*x);
end

F = F*Phi_i(x,X,i);
    
end

