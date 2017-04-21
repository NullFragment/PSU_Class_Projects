function [ F ] = F_vector( X )
%F_VECTOR Summary of this function goes here
%   Detailed explanation goes here
m = length(X);
F = zeros(m,1);

for i = 1:m-1
    F(i,1) = TwoPointRule(@f_x, i, i, X, X(i), X(i+1));
end

F(m,1) = TwoPointRule(@f_x, i, i, X, X(m-1), X(m));

end

