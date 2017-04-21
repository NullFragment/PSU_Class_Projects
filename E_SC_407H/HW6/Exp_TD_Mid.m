function [ Mid ] = Exp_TD_Mid( x, dx, step )
%Exp_TD_MID Summary of this function goes here
%   Detailed explanation goes here

i1 = (step+1/2)*dx;
i2 = (step-1/2)*dx;

Mid = -(D_x(x+i1) + D_x(x+i2) + (dx^2) - 1);

end

