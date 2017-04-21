function [ ] = Solver( Length, Steps, x0 )
%SOLVER Summary of this function goes here
%   Detailed explanation goes here
dx = (Length-x0)/Steps;
X = x0+dx:dx:Length;

for i = 1:Steps
    M(i) = C_exact(x0+i*dx);
end

K = TriDiag(X)
F = F_vector(X)

U = (K\F)'
M

end

