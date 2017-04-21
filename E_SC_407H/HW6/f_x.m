function [ F ] = f_x( x0, steps, stepsize )
%F_X is the given F(x) function

F = zeros(steps,1);

for i = 1:steps
    X = x0+(i*stepsize);
    if X > (pi/3)
        F(i,1) = 9.01*sin(3*X);
    else
        F(i,1) = 10*sin(3*X);
    end
end

end

