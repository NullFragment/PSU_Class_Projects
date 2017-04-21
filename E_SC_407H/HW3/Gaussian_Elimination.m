function [ B, x ] = Gaussian_Elimination( A, b )
%GAUSSIAN_ELIMINATION Summary of this function goes here
%   Detailed explanation goes here

%Forward Elimination
[B, c] = FWDElim(A,b);

%Backward Substitution
x = BWDSub(B, c);

end

