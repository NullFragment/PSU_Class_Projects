function [garbage] = Lagrange( func, x, xmin, xmax, order )
%LAGRANGE Generates a Lagrange polynomial for a given function.
garbage = 0;
h = (xmax - xmin)/order;
p = 0;
for i = 0:order
    L = 1;
    xcurr = xmin + i*h;
    for j = 0:order
        if j ~= i
            xj = xmin + j*h;
            L = L*((x - xj)/(xcurr - xj));
        end
    end
    p = p + func(xcurr)*L;
    M(i+1,2) = xcurr;
    M(i+1,1) = p;
end

plot(M(:,2),M(:,1))