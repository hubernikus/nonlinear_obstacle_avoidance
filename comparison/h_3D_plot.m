%% plot the desired path
figure; hold on; grid on; set(gcf,'color','w'); axis equal;
[x, y, z]=sphere;
h1=surf(x-2,y,z, 'FaceAlpha',0.2, 'EdgeColor', 'none', 'FaceColor', [0.8 0.5 0]);  % reactive area
radius = sqrt(0.28);
h2=surf(x*radius-2,y*radius,z*radius, 'FaceAlpha',0.2, 'EdgeColor', 'none', 'FaceColor', [0.8 0 0.5]);  % repulsive area
hold on;
h3=plot3(-2,0,0,'r.','MarkerSize',100);    % obstacle

th=0:0.1:2*pi;
h4=plot3(2*cos(th),2*sin(th),zeros(size(th)), 'r--', 'LineWidth', 2); % desired path

%% plot the traj
h5=plot3(p(:,1), p(:,2), p(:,3), 'm', 'LineWidth',2);
plot3(p(1,1), p(1,2), p(1,3), 'bo','MarkerSize', 8);
xlabel('X'); ylabel('Y'); zlabel('Z');
legend([h1, h2, h4, h5],'reactive surface','repulsive surface','desired path', 'trajectory');
view(39,23)
hold off;

%% plot the error
figure; hold on; grid on; set(gcf,'color','w')
time=e.Time; 
plot(e.Time, e.Data(:,1), 'LineWidth',2);
hold off;
