function [ A ] = MatrixGenerator( n )
%MATRIXGENERATOR Summary of this function goes here
%   Detailed explanation goes here

A = zeros(n:n);

for i = 1:n
    for j = 1:n
        A(i,j) = i^(j-1);
    end
end

end

