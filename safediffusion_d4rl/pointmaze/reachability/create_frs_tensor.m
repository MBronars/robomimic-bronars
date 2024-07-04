% create_frs_tensor.m
% This file creates the forward reachable set of the parameterized
% dynamics
%
% Wonsuhk Jung

clear; clc;

plot_on = false;

dim     = 7; % 1 (p_x) 2 (p_y) 3 (kvx) 4 (kvy) 5(kax) 6(kay) 7 (time)
t_plan  = 0.5;
t_total = 1;
dt      = 0.005;

% generates dynamics parameterized by K
[dyn_zero_to_t_plan, dyn_t_plan_to_t_total] = ...
    gen_traj_model(t_plan, t_total);

% get relative path and add path
currentFile = mfilename('fullpath');
gen_frs_path = fileparts(currentFile);
load_frs_path = fileparts(gen_frs_path);
addpath(fullfile(gen_frs_path,'traj_model'));
save_path = fullfile(load_frs_path,'frs_tensor_saved');

c_kv = 0;
c_ka = 0;
delta_kv = 5;
delta_ka = 7.5;
mean   = [c_kv, c_kv, c_ka, c_ka];
deltas = [delta_kv, delta_kv, delta_ka, delta_ka];

% create folder to save precomputed FRSs
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

% save vector of initial velocity subinterval ccenters
% save(fullfile(save_path, 'c_kvi.mat'), 'c_kvi');

% set options for reachability analysis:
options.timeStep = dt;
options.taylorTerms= 3; % number of taylor terms for reachable sets
options.zonotopeOrder= 2; % zonotope order... increase this for more complicated systems.
options.maxError = 1000*ones(dim, 1); % our zonotopes shouldn't be "splitting", so this term doesn't matter for now
options.verbose = false;
options.uTrans = 0; % we won't be using any inputs, as traj. params specify trajectories
params.U = zonotope([0, 0]);
options.advancedLinErrorComp = 0;
options.tensorOrder = 2;
options.reductionInterval = inf;
options.reductionTechnique = 'girard';
options.alg = 'lin';

% for numerical compactness
epsilon = 1e-6; 


% (1) FRS - first piece
params.tStart = 0;
params.tFinal = t_plan;

params.x0 = [0; 0; c_kv; c_kv; c_ka; c_ka; 0];
params.R0 = zonotope([params.x0, [0; 0; delta_kv; 0; 0; 0; 0], ...
                                [0; 0; 0; delta_kv; 0; 0; 0], ...
                                [0; 0; 0; 0; delta_ka; 0; 0], ...
                                [0; 0; 0; 0; 0; delta_ka; 0]]);

sys = nonlinearSys(dyn_zero_to_t_plan, dim, 1);

FRS_zero_to_t_plan = reach(sys, params, options);

% (2) FRS - second piece
t_plan_slice = FRS_zero_to_t_plan.timePoint.set{end};
params.R0 = t_plan_slice;
params.tStart = t_plan;
params.tFinal = t_total;

sys = nonlinearSys(dyn_t_plan_to_t_total, dim, 1);

FRS_t_plan_to_t_total = reach(sys, params, options);

FRS = [FRS_zero_to_t_plan.timeInterval.set; FRS_t_plan_to_t_total.timeInterval.set];

n_timestep = size(FRS, 1);
n_generator_max =  max(cellfun(@(x) size(x.deleteZeros().generators, 2), FRS));

FRS_tensor = zeros(n_timestep, n_generator_max, dim);

figure(1); clf; hold on; axis equal; grid on;
for i = 1:n_timestep
    
    if i == 1 || i == ceil(t_plan/dt)
        FRS{i} = FRS{i}.reduce('girard', 2);
        FRS{i} = FRS{i}.deleteZeros();
        Z = FRS{i}.Z;
    else
        FRS{i} = FRS{i}.deleteZeros();
        Z = FRS{i}.Z;
    end

    FRS_tensor(i, 1:size(Z, 2), :) = Z';

    if plot_on
        if i <= int(t_plan/dt)
            p = plot(FRS{i}, [1, 2], 'b', 'Filled', true);
            p.FaceAlpha = 0.05;
        else
            p = plot(FRS{i}, [1, 2], 'r', 'Filled', true);
            p.FaceAlpha = 0.05;
        end
    end
end
filename = fullfile(save_path, 'frs_tensor_mat.mat');
save(filename, 'FRS_tensor', 'options', 't_plan', 't_total', 'c_kv', 'c_ka', 'delta_kv', 'delta_ka', 'dt');


%% Testing overapproximation
palette = get_palette_colors;
%% checking the trajectory model
vx = 2;
vy = 5;
ax = 2;
ay = -5;
[t1, x1] = ode45(@(t, x) dyn_zero_to_t_plan(x), [0 t_plan], [0; 0; vx; vy; ax; ay; 0]);
[t2, x2] = ode45(@(t, x) dyn_t_plan_to_t_total(x), [t_plan t_total], x1(end, :));

figure; hold on; grid on;
x = [x1(:, 1); x2(:, 1)];
y = [x1(:, 2); x2(:, 2)];

for i = 1:n_timestep
    frs = FRS{i}.deleteZeros;
    frs_sliced = custom_slice(frs, [vx, vy, ax, ay], mean, deltas);
    p = plot(frs_sliced, [1, 2], 'b', 'Filled', true);
    p.FaceAlpha = 0.01;
end
p = plot(x, y); p.Color = palette.magenta; p.LineWidth = 2;
% xlim([-3, 3]); ylim([-3, 3]);

function new_Z = custom_slice(Z, val, mean, deltas)
determinates = (val - mean)./deltas;

c = Z.center;
G = Z.generators;
center = c + G(:, 1) * determinates(1) + ...
             G(:, 2) * determinates(2) + ...
             G(:, 3) * determinates(3) + ...
             G(:, 4) * determinates(4);
generators = G(:, 5:end);

new_Z = zonotope(center, generators);
end