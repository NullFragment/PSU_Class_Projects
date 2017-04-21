function [  ] = PDE_Solver( L, T )
%PDE_SOLVER Uses PDEPE to solve the partial differential equation we are
% given

m = 0;
x = linspace(0, L);
t = linspace(0, T);

sol = pdepe(m, @pde_p2, @pde_ic, @pde_bc, x, t);

u = sol(:,:,1);

surf(x,t,u)


end

