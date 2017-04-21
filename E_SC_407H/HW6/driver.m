function [  ] = driver( )
%DRIVER Runs the code for HW6

for i = 1:8
    figure(2^i)
    Solver(2^i, 0, pi/2)
end

PDE_Solver(pi/2, 10)




end

