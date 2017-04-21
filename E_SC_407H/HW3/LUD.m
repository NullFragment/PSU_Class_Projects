function [ A, L, U ] = LUD( A )
%LUD Summary of this function goes here
%   Detailed explanation goes here

% Check to see if matrix is square
[ i, j ] = size(A);
if (i ~= j)
    fprintf('The matrix is not square!\n');
    error('non-square matrix')
end

% Check to see if pivoting is necessary
for(x = 1:i)
    if A(x,x) == 0;
        error('pivoting neecesary');
    end
end

% Initialize L & U
L = eye(i);
U = A;  % Setting U = A allows us to perform the decomposition and keep 
        % the original matrix untouched.

for(x = 1:i)
    for(y = (x+1):i)
        L(y,x) = U(y,x)/U(x,x);
        for(z = 1:i)
            U(y,z) = U(y,z) - L(y,x)*U(x,z);
        end
    end
end

%Check to see if operation succeeded.
if (L*U ~= A)
    fprintf('LU Decomposition failed')
    error('LU Decomp. failure')
end

end