function [ L ] = Phi_i( x, X_points, i )
%Phi_i is the lagrange polynomial evaluated at Xi

L = 1;
m = length(X_points);

for j = 1:m
    if j ~= i
        L = L*(x-X_points(j))/(X_points(i) - X_points(j));
    end
end

end

