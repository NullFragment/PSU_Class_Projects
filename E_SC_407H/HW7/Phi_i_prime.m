function [ Sum ] = Phi_i_prime( x, X_points, i )
%PHI_I_PRIME Summary of this function goes here
%   Detailed explanation goes here

L = 1;
Sum = 0;
m = length(X_points);

for k = 1:m
    if k ~= i
       for j = 1:m
           if j ~= k
               if j ~= i
                   L =L*(x-X_points(j))/(X_points(i)-X_points(j));
               end
           end
       end
       Sum = Sum + L*(-X_points(k))/(X_points(i)-X_points(k));
       L=1;
    end
end