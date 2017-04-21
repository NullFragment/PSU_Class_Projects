function [x_n, x_n_1, x_n_2, x_n_a, x_n_b] = xn(iterations, filename)
if(nargin < 2)
    filename = 'xn';
end
x_n = 1;
x_n_1 = 1/4.1;
a = 15/4.1;
b = 14/16.81;
x_a = [];
x_b = [];
iter = [];
if(iterations >= 0)
    while(iterations >= 0)    
        if(iterations >= 2) 

            while(iterations >=2)
                x_n_2 = a*x_n_1 - b*x_n;
                x_n = x_n_1;
                x_n_1 = x_n_2;
                x_n_b = x_n_2;
                x_n_a = 1/power(4.1, iterations);
                iter(end+1) = iterations;
                x_a(end+1) = x_n_a;
                x_b = [x_n_b, x_b];
                iterations = iterations-1;


            end

        elseif(iterations == 1)
            x_n_b = 1/4.1;
            x_n_a = 1/power(4.1, iterations);
            iter(end+1) = iterations;
            x_a(end+1) = x_n_a;
            x_b(end+1) = x_n_b;
            iterations = iterations - 1;

        elseif(iterations == 0)
            x_n_b = 1;
            x_n_a = 1/power(4.1, iterations);
            iter(end+1) = iterations;
            x_a(end+1) = x_n_a;
            x_b(end+1) = x_n_b;
            iterations = iterations - 1;
        end
    end
    
    x_a
    x_b
    xn_plot = plot(iter, x_a);
    hold all;
    xn_plot = plot(iter, x_b);
    file = strcat(filename, '.jpg');
    saveas(xn_plot, file);
elseif (iterations < 0)
        error('N must not be negative');
end



