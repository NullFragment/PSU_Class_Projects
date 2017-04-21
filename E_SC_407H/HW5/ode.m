function [ vals ] = ode( t )
%ODE Summary of this function goes here
%   Detailed explanation goes here
vals = zeros(1,2);
m = 32.2/2;
theta = (pi/4)*cos((sqrt(m)*t));
v = (-pi/4)*sqrt(m)*sin(sqrt(m)*t);

vals(1,1) = theta;
vals(1,2) = v;
end

