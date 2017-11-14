clear;clc
x = csvread ('comm_times.csv');
for i = 10:10:90
  idx = ( x(:,3)==i );
  x_new = x(idx,:);
  tx = x_new(:,1);
  ty = x_new(:,2);
  tz = x_new(:,4);
  tri = delaunay(tx,ty);
  plt_idx = i/10;
  subplot(3,3,plt_idx);
  h = trisurf(tri, tx, ty, tz);
  axis([0 25 0 10000 0 500]);
  t = strcat("Time Steps = ",num2str(i));
  title(t);
  xlabel ("processors");
  ylabel ("grid size");
  zlabel ("time");
  f = "comm_time.png"
end
print(f,"-S1920,1080","-dpng")

clear;clc
x = csvread ('parallel_times.csv');
for i = 10:10:90
  idx = ( x(:,3)==i );
  x_new = x(idx,:);
  tx = x_new(:,1);
  ty = x_new(:,2);
  tz = x_new(:,4);
  tri = delaunay(tx,ty);
    plt_idx = i/10;
  subplot(3,3,plt_idx);
  h = trisurf(tri, tx, ty, tz);
  t = strcat("Time Steps = ",num2str(i));
  title(t);
  xlabel ("processors");
  ylabel ("grid size");
  zlabel ("time");
  f = strcat("parallel_time_steps_",num2str(i));
  f = strcat(f,".png");
  print (f, "-dpng");
end




%% MESH FULL
%clear;clc
%x = csvread ('test.csv');
%tx = x(:,1);
%ty = x(:,2);
%tz = x(:,4);
%[zz] = meshgrid(tz);
%mesh (tx, ty, zz);
%xlabel ("processors");
%ylabel ("steps");
%zlabel ("time");

%% MESH SEGMENT
%clear;clc
%x = csvread ('test.csv');
%idx = ( x(:,3)==10 );
%x_new = x(idx,:);
%tx = x_new(:,1);
%ty = x_new(:,2);
%tz = x_new(:,4);
%[tz] = meshgrid(tz);
%mesh(tx,ty,tz);
%xlabel ("processors");
%ylabel ("steps");
%zlabel ("time");

%% CONTOUR
%clear;clc
%x = csvread ('test.csv');
%x(:,[3]) = [];
%contour3(x);
%xlabel ("processors");
%ylabel ("steps");
%zlabel ("time");

%% SCATTER
%clear;clc
%x = csvread ('test.csv');
%idx = ( x(:,3)==10 );
%x_new = x(idx,:);
%tx = x_new(:,1);
%ty = x_new(:,2);
%tz = x_new(:,4);
%scatter3(tx,ty,tz,'filled')
%xlabel ("processors");
%ylabel ("steps");
%zlabel ("time");

%% Triangle Surface
%clear;clc
%x = csvread ('test.csv');
%tx = x(:,1);
%ty = x(:,3);
%tz = x(:,4);
%tri = delaunay(tx,ty);
%h = trisurf(tri, tx, ty, tz);
%xlabel ("processors");
%ylabel ("grid size");
%zlabel ("time");