function [ B, c ] = FWDElim( A, b )
%FWDELIM Performs naive Gaussian forward elimination given a matrix and
%its solution vector.

[ m, n ] = size(A);

B = A; %preserve original matrix and return augmented matrix;
c = b; %preserve original solution vector and return augmented solutions;

for i = 1:m
    for j = i+1:n
        divi = B(j,i)/B(i,i);
        B(j,i) = B(j,i) - B(i,i)*divi;
        for(k=i+1:n)
            B(j,k) = B(j,k) - divi*B(i,k);
        end
        c(j) = c(j) - divi*c(i);
    end
end
    
