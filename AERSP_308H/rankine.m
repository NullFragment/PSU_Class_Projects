function [  ] = rankine( K, Uinf, RG, n )
%% creates a potential Flow Field for various functions
%plotting limits 
xmin=6;xmax=-xmin;dx=(xmax-xmin)/200;
%create domain of interest (Cart)
[X,Y]=meshgrid(xmin:dx:xmax);
%Polar Coordinates
R=sqrt(X.^2+Y.^2); 
TH=atan2M(Y,X);

%velocity Field initialization
u=0*R;
v=0*R;

%setup plotting
numContour=20; %number of contours 
figure(1); %be sure to start with a clean figure

EPS=1.e-5; % this removes log(0) issues 

%Fluid properties
pinf=1e5; %Pa
rhoinf= 1; %kg/m^3


%%%Lets start adding components

%Uniform flow (adjust Uinf to modify strength)
alpha= 0; %incidence angle in rad
PHI = Uinf*(Y*cos(alpha)+X*sin(alpha));
PSI = Uinf*(X*cos(alpha)-Y*sin(alpha));
u=Uinf*cos(alpha);
v=Uinf*sin(alpha);

% Doublet
X0=0;
Y0=.0;
THS=atan2M((Y-Y0),(X-X0));
RS=sqrt((X-X0).^2+(Y-Y0).^2+EPS);
lambda = Uinf*RG^2;
PHI = PHI - (lambda.*sin(THS))./RS;
PSI = PSI + (lambda.*cos(THS))./RS;
u = u + (lambda .* cos(2.*THS)./RS);
v = v - (lambda .* sin(2.*THS)./RS);

%vortex (adjust K to modify strength)s
X0=0;
Y0=.0;
THS=atan2M((Y-Y0),(X-X0));
RS=sqrt((X-X0).^2+(Y-Y0).^2+EPS);
PHI = -K*log(RS) + PHI;
PSI = K*THS + PSI;
u=u+K*sin(THS)./RS;
v=v+K*cos(THS)./RS;


%Bernoulli Eqn
Vmag=sqrt(u.^2+v.^2); %calculate velocity magnitude 
Vmag=min(Vmag,1.64*Uinf); %limit to realistic values, eliminates peaks inside body
p=0.5*rhoinf*(Uinf^2-Vmag.^2); %apply Bernoulli to get pressure field

subplot(2,2,n);
%Plot %
CS=contourf(X,Y,Vmag,numContour*2,':','LineWidth',0.01); %filled contour of Vel Mag%CS=contourf(X,Y,p,numContour*2,':','LineWidth',0.01); %filled contour of pressure

hold on
CS=contour(X,Y,PHI,numContour,'k','LineWidth',2);
%CS=contour(X,Y,PSI,numContour,'--b','LineWidth',2);
xlabel('X-direction')
ylabel('Y-direction')
hold off
end 