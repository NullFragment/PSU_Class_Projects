function [ x ] = BWDSub( B, b)
%BWDSUB performs backward substitution on a matrix processed via forward
%Gaussian elimination
[m, n] = size(B);

C = B;
x = zeros(n,1);

x(n)=b(n)/B(n,n);
for i=n-1:-1:1
    sum = 0;
    for j=i+1:n
        sum = sum + B(i,j)*x(j);
    end
    x(i)=(b(i)-sum)/B(i,i);
end

end

