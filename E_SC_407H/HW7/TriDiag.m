function [ K ] = TriDiag( X )
%TRIDIAG Summary of this function goes here
%   Detailed explanation goes here
steps = length(X);
K = zeros(steps);

for i = 1:steps-1
    K(i,i+1) = TwoPointRule(@Coeff, i, i+1, X, X(i), X(i+1));
    K(i, i) = TwoPointRule(@Coeff, i, i, X, X(i), X(i+1));
    K(i+1,i) = TwoPointRule(@Coeff, i+1, i, X, X(i), X(i+1));
end

K(steps,steps) = K(steps-1, steps-1);

end

