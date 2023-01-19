function vec = evaluate_field_with_six_obstacles(pos)
% function evaluate field

% TODO: some parameters should be given as input - to allow
% generalization / faster evaluation
% But for now, the largest slow-down is the usage of symbolic math...

% if size(pos, 1)==1 && size(pos, 2) > 1
%     % Transpose to be consisten
%     pos = pos';
% end

% Parameters
l1 = 0.1; l2 = 0.1;

%%%% draw the bump function boundary
c1 = -0.72;
c2 = -0.72;
c3 = -0.2;
c4 = -0.2;
c5 = -0.23;

R = 2;
a = 1; b = 0.5;
beta = pi/4;

syms x y
phi = x^2 + y^2 -R^2;
vphi1 = x^2/a^2 + (y+R)^2/b^2 - 1;
vphi2 = ((x-R)*cos(beta)+y*sin(beta))^2/a^2 + ((x-R)*sin(beta)-y*cos(beta))^2/b^2 - 1;
vphi3 = (x+2.6)^2/b^2 + y^2/a^2 - 1;
vphi4 = (x+1.4)^2/b^2 + y^2/a^2 - 1;
vphi5 = ((x-0.9)^2+(y-R)^2)*((x+0.9)^2+(y-R)^2) - 0.9;

%%% calculate the vector field
vfphi = vf(phi, 1);
vfvphi1 = vf(vphi1, 1);
vfvphi2 = vf(vphi2, 1);
vfvphi3 = vf(vphi3, 1);
vfvphi4 = vf(vphi4, 1);
vfvphi5 = vf(vphi5, 1);


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

vec = vec';
end

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