clc; clear all; close all;
% All path m7erging are specifical for linux / unix

%datapath = "/home/lukas/Code/nonlinear_avoidance/comparison/data";
datapath = "/home/lukas/Code/nonlinear_obstacle_avoidance/nonlinear_avoidance/comparison/data";
inputfile = "initial_positions.csv";

outputfolder = "guiding_field";

% Create Outputfolder if needed
output_path = datapath + "/" + outputfolder;
mkdir(output_path);

% Import values from files
config_file = "comparison_parameters.json";
config_path = datapath+ "/../" + config_file;
config_str = fileread(config_path);
config_values = jsondecode(config_str);

% Set variables and import files

start_positions = readtable(datapath + "/" + inputfile,...
    'NumHeaderLines', 1);

dimension = size(start_positions, 2);
it_max = config_values.it_max
dt = config_values.delta_time
conv_margin = config_values.conv_margin

t_max = it_max * dt;
% t_span=0:dt:t_max;

% paramStruct.StartTime = num2str(0);
% paramStruct.StopTime = num2str(t_max);

%% Evaluate trajectories
for pp = 1: size(start_positions, 1)

%     if pp > 5
%         break
%     end
    fprintf("Trajectory %d / %d \n", pp, size(start_positions, 1));

    pos0 = table2array(start_positions(pp, :));
    
    model_name = 'd_mixvf_sim';
    open_system(model_name);
    mdlWks = get_param('d_mixvf_sim','ModelWorkspace');
    assignin(mdlWks,'initial_position', pos0)
%     simIn = Simulink.SimulationInput(model_name);

    % Simulink solver
%     sim('d_mixvf_sim', t_max);
    out = sim(model_name);

%     positions = zeros(it_max, dimension);
%     positions(1, :) = table2array(start_positions(pp, :));%     for ix = 2: it_max
%         % No convergence calculation possible
%         
%         velocity = evaluate_field_with_six_obstacles(positions(ix-1, :));
% 
%         if norm(velocity) < conv_margin
%             positions = positions(1:ix, :); 
%             fprintf("Converged at it=%d.\n", ix)
%             break
%         end
% 
%         % Normalize velocity to be consisten across algorithms
%         velocity = velocity / norm(velocity);
%    
%         positions(ix, :) = velocity * dt + positions(ix -1, :);
%     end
%     fprintf("Stopping after max_it=%d.\n", ix)
    
    positions = p;
    

    % Start file numbering at 0
    filename = sprintf( 'trajectory%03d.csv', pp-1);
    writematrix(positions, output_path + "/" + filename);
end

fprintf("Finished evaluation.\n")
writematrix(positions, output_path + "/" + filename);


%% Integrate desired trajectories
start_position = [-0.966468219, 2.43618389];
model_name = 'd_mixvf_sim';
open_system(model_name);
mdlWks = get_param('d_mixvf_sim','ModelWorkspace');
assignin(mdlWks,'initial_position', start_position);
out = sim(model_name, 20.0);
positions = p;

filename = sprintf( outputfolder + '_cycle.csv', pp-1);
writematrix(positions, datapath + "/" + filename);

fprintf("Finished evaluation \n.")
