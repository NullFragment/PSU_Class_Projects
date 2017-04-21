function [ Coeff ] = Coeff( x, X, i, j )
%COEFF Summary of this function goes here
%   Detailed explanation goes here
if x > pi/3
    E = 100;
else
    E = 1;
end

A = Phi_i_prime(x, X, i);
B = Phi_i_prime(x, X, j);
C = Phi_i(x, X, i);
D = Phi_i(x, X, j);
Coeff = A*B*E - C*D;
end

