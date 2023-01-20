clear all; close all; clc;

%% plot the desired path
%%%% plot two desired paths that are used to generate vfs
R = 2;
a = 1; b = 0.5;
beta = pi/4;

figure; hold on; grid on; set(gcf,'color','w'); axis equal;
syms x y
phi = x^2 + y^2 -R^2;
fimplicit(phi, 'Color', 'red', 'LineWidth', 2)
plot(0, 0, 'Marker','.','MarkerSize',10);

vphi1 = x^2/a^2 + (y+R)^2/b^2 - 1;
fimplicit(vphi1, 'LineWidth', 2)
plot(0, -R, 'Marker','.','MarkerSize',10);

vphi2 = ((x-R)*cos(beta)+y*sin(beta))^2/a^2 + ((x-R)*sin(beta)-y*cos(beta))^2/b^2 - 1;
fimplicit(vphi2, 'LineWidth', 2)
plot(R, 0, 'Marker','.','MarkerSize',10);

vphi3 = (x+2.6)^2/b^2 + y^2/a^2 - 1;
fimplicit(vphi3, 'LineWidth', 2)
plot(-2.6, 0, 'Marker','.','MarkerSize',10);

vphi4 = (x+1.4)^2/b^2 + y^2/a^2 - 1;
fimplicit(vphi4, 'LineWidth', 2)
plot(-1.4, -0, 'Marker','.','MarkerSize',10);

vphi5 = ((x-0.9)^2+(y-R)^2)*((x+0.9)^2+(y-R)^2) - 0.9;
fimplicit(vphi5, 'LineWidth', 2)
plot(0, R, 'Marker','.','MarkerSize',10);
plot(0.9, R, 'Marker','.','MarkerSize',10);
plot(-0.9, R, 'Marker','.','MarkerSize',10);
ylim([-2.6 2.7])

%%%% draw the bump function boundary
c1 = -0.72;
vphi1_b = vphi1 - c1;
fimplicit(vphi1_b, 'LineStyle', '-.', 'LineWidth', 2);

c2 = -0.72;
vphi2_b = vphi2 - c2;
fimplicit(vphi2_b, 'LineStyle', '-.', 'LineWidth', 2);

c3 = -0.2;
vphi3_b = vphi3 - c3;
fimplicit(vphi3_b, 'LineStyle', '-.', 'LineWidth', 2);

c4 = -0.2;
vphi4_b = vphi4 - c4;
fimplicit(vphi4_b, 'LineStyle', '-.', 'LineWidth', 2);

c5 = -0.23;
vphi5_b = vphi5 - c5;
fimplicit(vphi5_b, 'LineStyle', '-.', 'LineWidth', 2);

%% plot the traj

pos0 = [2.0, 0.8];
sim('d_mixvf_sim', 5)


t=1:length(e.Time);
plot(p(t,1),p(t,2),'m', 'LineWidth', 2);
xlabel('x'); ylabel('y');

disp("Simulation done.")

%% plot the vector field
l1 = 0.1; l2 = 0.1;
step = 0.5; 
xx = -3.5 : step : 3.5; 
yy = -3 : step : 3;
[X, Y] = meshgrid(xx,yy);
VX = zeros(size(X));
VY = zeros(size(X));

%%% calculate the vector field
vfphi = vf(phi, 1);
vfvphi1 = vf(vphi1, 1);
vfvphi2 = vf(vphi2, 1);
vfvphi3 = vf(vphi3, 1);
vfvphi4 = vf(vphi4, 1);
vfvphi5 = vf(vphi5, 1);

for i = 1:size(X,1)
    fprintf("Column: %d / %d \n", i, size(X,1));
    for j = 1:size(X,2)
        pos = [X(i,j) Y(i,j)];

        n_vfphi = subs(vfphi, [x y], pos);
        n_vfvphi1 = subs(vfvphi1, [x y], pos);
        n_vfvphi2 = subs(vfvphi2, [x y], pos);
        n_vfvphi3 = subs(vfvphi3, [x y], pos);
        n_vfvphi4 = subs(vfvphi4, [x y], pos);
        n_vfvphi5 = subs(vfvphi5, [x y], pos);
        
        n_vphi1 = subs(vphi1, [x y], pos);
        n_vphi2 = subs(vphi2, [x y], pos);
        n_vphi3 = subs(vphi3, [x y], pos);
        n_vphi4 = subs(vphi4, [x y], pos);
        n_vphi5 = subs(vphi5, [x y], pos);
        
        kq1 = bf(1, n_vphi1, c1, l1, l2);
        kq2 = bf(1, n_vphi2, c2, l1, l2);
        kq3 = bf(1, n_vphi3, c3, l1, l2);
        kq4 = bf(1, n_vphi4, c4, l1, l2);
        kq5 = bf(1, n_vphi5, c5, l1, l2);
        
        kr1 = bf(2, n_vphi1, c1, l1, l2);
        kr2 = bf(2, n_vphi2, c2, l1, l2);
        kr3 = bf(2, n_vphi3, c3, l1, l2);
        kr4 = bf(2, n_vphi4, c4, l1, l2);
        kr5 = bf(2, n_vphi5, c5, l1, l2);
        
        vec = kq1*kq2*kq3*kq4*kq5 * n_vfphi / norm(n_vfphi) + ...
              kr1 * n_vfvphi1 / norm(n_vfvphi1) + ...
              kr2 * n_vfvphi2 / norm(n_vfvphi2) + ...
              kr3 * n_vfvphi3 / norm(n_vfvphi3) + ...
              kr4 * n_vfvphi4 / norm(n_vfvphi4) + ...
              kr5 * n_vfvphi5 / norm(n_vfvphi5);


        %vec = vec / norm(vec);
        VX(i,j) = vec(1);
        VY(i,j) = vec(2);
    end
end

%%% plot the vector field
quiver(X,Y,VX, VY,'b');
hold off;


%% plot the vector field - using function
l1 = 0.1; l2 = 0.1;
step = 0.5; 
xx = -3.5 : step : 3.5; 
yy = -3 : step : 3;
[X, Y] = meshgrid(xx,yy);
VX = zeros(size(X));
VY = zeros(size(X));

%%% calculate the vector field
vfphi = vf(phi, 1);
vfvphi1 = vf(vphi1, 1);
vfvphi2 = vf(vphi2, 1);
vfvphi3 = vf(vphi3, 1);
vfvphi4 = vf(vphi4, 1);
vfvphi5 = vf(vphi5, 1);

for i = 1:size(X,1)
    fprintf("Column: %d / %d \n", i, size(X,1));
    for j = 1:size(X,2)
        pos = [X(i,j) Y(i,j)];
        
        vec = evaluate_field_with_five_obstacles(pos);

        %vec = vec / norm(vec);
        VX(i,j) = vec(1);
        VY(i,j) = vec(2);
    end
end

%%% plot the vector field
quiver(X,Y,VX, VY,'b');
hold off;
%% plot the error
figure; hold on; grid on; set(gcf,'color','w')
time=e.Time(t); e1=e.Data((t),1); e2=e.Data((t),2); 
e_norm=sqrt(e1.^2+e2.^2);
plot(time, e1, 'LineWidth',2);
% plot(time, e2, 'LineWidth',2);
% plot(time, e_norm, 'LineWidth',2, 'LineStyle', '--');
plot(time, c1*ones(size(time)), 'LineWidth',1, 'LineStyle', '--');
legend('e1', 'safety bound');
% legend('e1', 'e2', 'safety bound');
%legend('e1', 'e2', '||e||', 'safety bound');
hold off;




%% function to creathe vector field
function out = vf(phi, k)
    syms x y
    E = [0, -1; 1 0];
    gra = [diff(phi,x); diff(phi,y)];
    out = E * gra - k * phi * gra;
end

%% bump function
function out = bf(type, phi, thres, l1, l2)
    if( phi < thres )
        k = 0;
    else
        if( phi < 0 )
            f1 = exp( l1/(thres - phi) );
            f2 = exp( l2/(phi - 0) );
            k = f1/(f1+f2);
        else
            k = 1;
        end
    end
        
    if(type == 1)   % zero-inside bump function
        out = k;
    elseif(type == 2)  % zero-outside bump function
        out = 1 - k;
    end
end
