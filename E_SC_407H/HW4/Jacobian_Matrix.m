function [ J ] = Jacobian_Matrix( u, v )
%JACOBIAN_MATRIX generates values for the jacobian of problem 2
syms a b;
equations = [((a - a^2/2) - a*b); ((a-1)*b)];
J = jacobian(equations);
J = subs(J, [a b], [u v]);

end
