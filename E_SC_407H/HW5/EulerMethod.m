function [ Values, t ] = EulerMethod( func, f0, t_init, t_fin, stepsize )
%EULERMETHOD Summary of this function goes here
%   Detailed explanation goes here

t = [t_init:stepsize:t_fin]';
Values = f0;
steps = (t_fin-t_init)/stepsize;

for i = 1:steps-1
    Values(i+1,:) = Values(i,:) + (stepsize*i)*func(t(i+1));
end

end

