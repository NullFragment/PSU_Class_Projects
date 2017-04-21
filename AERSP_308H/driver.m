clear;clc;

%% Plot Stream Functions
for i = 1:4
    rankine(i-1,1,1,i);
end


%% Plot Velocity and Pressure along body
th = 0:pi/100:2*pi;
X = meshgrid(sin(th));
Y = meshgrid(cos(th));
m = 5;
pos = 1;
for K = 0:1
    b = X.*0;
    v = -2.*X + K;
    p = 500*(1 - 4.*X^2 + 4*K.*X - K^2);
    figure(m)
    subplot(2,2,pos)
    quiver(X,Y,v,b,2)
    axis([ -1 1 -1 1 ])
    str = strcat('Velocity, K = ', num2str(K));    title(str)
    figure(m)
    pos = pos+1;
    subplot(2,2,pos)
    quiver(X,Y,p,b,2)
    axis([ -1 1 -1 1 ])
    str = strcat('Pressure, K = ', num2str(K));
    title(str)
    pos = pos + 1;
end



%% Calculate Coefficients of Drag and Lift
% for i = 0:1
%     D = @(theta) drag(theta, i);
%     L = @(theta) lift(theta, i);
%     Drag = -integral(D, 0, 2*pi);
%     Lift = -integral(L, 0, 2*pi);
%     M(i+1, 1) = i;
%     M(i+1, 2) = Lift/(0.5*1000*1*1*1*2);
%     M(i+1, 3) = Drag/(0.5*1000*1*1*1*2);
% end 
% M = dataset({M 'K' 'Cl' 'Cd'})