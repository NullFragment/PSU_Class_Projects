function [ Coeff ] = TriDiag( Top, Mid, Bot, steps, x0, stepsize )
%TRIDIAG Creates a tridiagonal matrix given a function for the top,
% middle and bottom diagonals, however the last spot is modified for the
% problem.

Coeff = zeros(steps);

for i = 1:steps-1
    dx_top = (i-1/2)*stepsize;
    dx_bot = (i+1/2)*stepsize;
    Coeff(i,i+1) = Top(x0+dx_top);
    Coeff(i, i) = Mid(x0, stepsize, i);
    Coeff(i+1,i) = Bot(x0+dx_bot);
end

dx = (steps-(3/2))*stepsize;
Coeff(steps,steps) = Bot(x0+dx)+stepsize^2;

end

