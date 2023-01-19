clc; clear all; close all;

datapath = "/home/lukas/Code/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance/rotational/comparison/data";
inputfile = "initial_positions.csv";

outputfolder = "guiding_field";

%% Create Outputfolder if needed
output_path = datapath + "/" + outputfolder;
mkdir(output_path);

%% Evaluate trajectories

% The path adition is specifically for linux / unix
start_positions = readtable(datapath + "/" + inputfile,...
    'NumHeaderLines',3);

dimension = size(start_positions, 2);
it_max = 1000
dt = 0.01
conv_margin = 1e-3

for pp = 1: size(start_positions, 1)
    fprintf("Trajectory %d / %d \n", pp, size(start_positions, 1));
    positions = zeros(it_max, dimension);
    positions(1, :) = table2array(start_positions(pp, :));

    for ix = 2: it_max
        % No convergence calculation possible
        velocity = evaluate_field_with_six_obstacles(positions(ix-1, :));
        
        positions(ix, :) = velocity * dt + positions(ix -1, :);

        if norm(velocity) < conv_margin
            positions = positions(1:ix, :); 
            fprintf("Converged at it=%d.\n", ix)
            break
        end
    end
    fprintf("Stopping after max_it=%d.\n", ix)
    filename = sprintf( 'trajectory%03d.csv', pp);
    writematrix(positions, output_path + "/" + filename);
end

fprintf("Finished evaluation.\n")