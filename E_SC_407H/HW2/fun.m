function [y] = fun(R)

L = 5;
C = 10^-4;
t = 0.05;
qq0 = 0.01;


sq = sqrt([1/(L*C)] - [R/(2*L)]^2);
y = exp(-[R*t]/[2*L]) * cos(sq * t) - qq0;