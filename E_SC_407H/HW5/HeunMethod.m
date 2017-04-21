function [ Values, t ] = HeunMethod( func, f0, t_init, t_fin, stepsize )
%HEUNMETHOD Summary of this function goes here
%   Detailed explanation goes here

t = [t_init:stepsize:t_fin]';
Values = f0;
steps = (t_fin-t_init)/stepsize;
Values(2,:) = Values(1,:) + stepsize*func(t(2));

for i = 1:steps-1
    j = i+1;
    m = (stepsize*i)/2;
    
    Values(j,:) = Values(1,:) + m*(func(t(i)) + func(t(j)));
end

end

