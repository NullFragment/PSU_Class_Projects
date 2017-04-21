function [ ] = Solver(steps, x0, L)
%SOLVER Solves the time-independent PDE given the number of steps,
% the initial starting position and the length of the bar.

dx = ((L)-x0)/steps;
div = (dx^2);

X = (x0+dx):dx:L;

for i = 1:steps
    M(i) = C_exact(x0+i*dx);
end

D = TriDiag(@D_x,@TD_Mid,@D_x, steps, x0, dx);

D = (1/div).*D;
F = -f_x(x0,steps,dx);
C = (D\F)';

plot(X,C,X,M)
hold all
legend('Numeric', 'Exact')
hold all


end
