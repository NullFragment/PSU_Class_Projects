function [ eig ] = QR_Method( A, max_iter, tol )
%QR_METHOD - finds the eigenvalues of a given matrix using the QR method
error = 999;
A_new = A;
iter = 1;
while (iter <= max_iter && error > tol)
    A_old = A_new;
    [Q R] = myQR(A_old);
    A_new = inv(Q)*A*Q;
    error = abs((norm(A_old)-norm(A_new))/norm(A_old));
    iter = iter+1;
end

eig = diag(A_new);

end

