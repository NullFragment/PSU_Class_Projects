function [  ] = Implicit_Solution( steps, L, x0, T, dt )
%IMPLICIT_SOLUTION Summary of this function goes here
%   Detailed explanation goes here

dx = (L-x0)/steps;
t = 0:dt:T;
x = 0:dx:L;
m = length(t);

mult = -(dt/(dx^2));
for i = 1:m
    D = TriDiag(@D_x,@Imp_TD_Mid,@D_x, steps, x0, dx);
    D = mult.*D;
    
    F = -f_x(x0,steps,dx);
    
    A = F\D;
    A
end

Sol

end

