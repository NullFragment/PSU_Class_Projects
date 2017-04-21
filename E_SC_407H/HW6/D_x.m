function [ Dx ] = D_x( x )
%D_X D(x) function

if x > (pi/3)
    Dx = 100;
else
    Dx = 1;
end

end

