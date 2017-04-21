function [ Values, t ] = RK4Method( func, f0, t_init, t_fin, stepsize )
%RK4Method Summary of this function goes here
%   Detailed explanation goes here
m = stepsize;
t = [t_init:m:t_fin]';
Values = f0;
steps = (t_fin-t_init)/m;

for i = 1:steps-1
    k1 = m*func(t(i));
    k2 = m*func(t(i) + m/2);
    k3 = m*func(t(i+1));
    
    Values(i+1,:) = Values(1,:) + (k1+4*k2+k3)/6;
end

end

