function [ X, Y, LSolved, USolved ] = LU_Solver( L, U, B )
%LU_SOLVER Summary of this function goes here
%   Detailed explanation goes here
[ m, n ] = size(L);

X = zeros(m,1);
Y = zeros(m,1);

[LSolved, Y] = Gaussian_Elimination(L,B);
[USolved, X] = Gaussian_Elimination(U,Y);

