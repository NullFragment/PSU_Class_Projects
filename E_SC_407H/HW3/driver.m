fprintf('Problem 1:\n\n\n')
A = MatrixGenerator(6);
B = [106.4; 57.79; 32.9; 19.52; 12.03; 7.67];
[A, L, U] = LUD(A)
[X, Y, LSolved, USolved] = LU_Solver(L, U, B)


fprintf('\n\n\nProblem 2:\n\n\n')
syms x y z;
NR = [ 3*x + 4*y - z - 3; x - 4*y + 2*z + 1; -2*x - y + 5*z - 2];
JC = [ 3*x, 4*y, -z; x, -4*y, 2*z; -2*x, -y, 5*z];
b = [ 3; -1; 2 ];
x0 = [0;0;0];
relaxation = .5;

[NRsol, NRiterations] = Newton_Raphson_SOE(NR, x0, 20, 10e-8)
[JCsol, JCiterations] = Jacobi_Method(JC, b, x0, 100, 10e-8)
[SORsol, SORiterations] = SOR(JC, b, x0, relaxation, 100, 10e-8)