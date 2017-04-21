clear,clc;

h = 1
i = 2
j = 3
k = 4

a1 = h + i
a2 = j - k
a3 = h * j
a4 = i / k
 
b = h*(i/j - k)

x = 0.37128
c1 = sin(x)
c2 = cos(x)
c3 = tan(x)
c4 = exp(x)
c5 = log(x)


A = [ 1, 2, 3 ]
B = [ 4; 5; 6 ]

C = A/7

D = rand(1, 5)

E = [ 50, 22, 34; 41, 56, 64; 37, 48, 69 ]
F = E'
G = det(E)
H = inv(E)
I = E * B
J = dot( E, E )
K = pi * E
L = E(2, 2)
M1 = min(min(E))
M2 = max(max (E))

aa = 0;
for n = 1:50
    aa = aa +((-1)^(n+1))/(2*n - 1);
end
aa
bb = 0; 
n = 1;
while n < 51
    bb = bb +((-1)^(n+1))/(2*n - 1);
    n = n+1;
end
bb
phi = bb
epsilon = (pi - phi)/pi

lims = 0:.0001:4*pi;
f = sin(2*pi*lims);
g = cos(2*pi*lims);
xlabel('Radians');
ylabel('Sinusoid  Values');
fplot = plot(f);
hold all
gplot = plot(g);
legend([fplot, 'sine'], [gplot,'cosine']);

print -djpeg test.jpg