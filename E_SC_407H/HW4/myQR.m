function [ Q R ] = myQR( A )
%QR Decomposes a matrix into Q and R components

[m n] = size(A);
if m ~= n
    error('Matrix is not square.')
end

Q = zeros(m);
R = zeros(m);

%Initialize Q's first row for recursion
u = A(:,1);
q = -u/norm(u);
Q(:,1) = q;

%Create matrix Q
for i = 2:n
  v = A(:,i);
  dotp = 0;
  for j = 1:i-1
      q = Q(:,j);
      dotp = dotp + dot(v,q)*q;
  end
  u = v - dotp;
  q = u/norm(u);
  Q(:,i) = q;
end

%Create matrix R
for k = 1:n
    e = Q(:,k);
    for l = 1:n
        R(k,l) = dot(A(l,:),e);
    end
end
  
end

