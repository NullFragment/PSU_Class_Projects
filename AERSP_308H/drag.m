function [ D ] = drag( theta, K )
P = 1/2*(1000)*(1-4*(sin(theta)).^2+4*K*sin(theta) - K^2);
D = P.*cos(theta);
end

