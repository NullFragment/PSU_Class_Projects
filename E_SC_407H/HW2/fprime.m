function [y] = fprime(R)

L = 5;
C = 10^-4;
t = 0.05;

sq = sqrt([1/(L*C)] - [R/(2*L)]^2);
e = exp(-[(t*R)/(2*L)]);

y = -[t/(2*L)] * e * cos(sq * t) + [(R*t)/(4*(L^2)*sq)] * e * sin(sq*t);