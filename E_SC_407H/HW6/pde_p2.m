function [ c,f,s ] = pde_p2( x, t, u, DuDx )
%PDE_P2 This is the PDE of part 2

if x > pi/3
    f = 100*DuDx;
    s = -u+9.01*sin(3*x);
else
    f = DuDx;
    s = -u+10*sin(3*x);
end

c = 1;

end

