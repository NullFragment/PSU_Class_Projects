fprintf('*************************************\nProblem 1\n')
fprintf('*************************************\nx=3.14\n')

fprintf('*************************************\nOrder 3\n')
Lagrange(@mysinc, 3.14159, 0, pi, 3)
hold all
fprintf('*************************************\nOrder 6\n')
Lagrange(@mysinc, 3.14159, 0, pi, 6)
hold all
fprintf('*************************************\nOrder 9\n')
Lagrange(@mysinc, 3.14159, 0, pi, 9)

fprintf('*************************************\nProblem 2\n')
fprintf('*************************************\nValues u = v = 0')
J = Jacobian_Matrix(0,0)
calc_eigs = QR_Method(J, 20, 10^-16)
true_eigs = eig(J)

fprintf('\n*************************************\nValues u =2, v = 0')
J = Jacobian_Matrix(2,0)
calc_eigs = QR_Method(J, 20, 10^-16)
true_eigs = eig(J)


fprintf('\n*************************************\nValues u = 1 v = 1/2')
J = Jacobian_Matrix(1,1/2)
calc_eigs = QR_Method(J, 20, 10^-16)
true_eigs = eig(J)

fprintf('\n*************************************\n')