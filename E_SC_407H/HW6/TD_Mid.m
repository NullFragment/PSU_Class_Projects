function [ Mid ] = TD_Mid( x, dx, step )
%TD_MID This term is the coefficient of C_i, or the middle row of the
% tridiagonal matrix.

i1 = (step+1/2)*dx;
i2 = (step-1/2)*dx;

Mid = -(D_x(x+i1) + D_x(x+i2) + (dx^2));

end

