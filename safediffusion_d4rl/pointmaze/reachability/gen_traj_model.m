function [dyn_zero_to_t_plan, dyn_t_plan_to_t_total] = gen_traj_model(t_plan,t_total)
%GENERATE_PARAMETERIZED_DYNAMICS prepares dynamics for CORA 2020 with the
%particular parameterization described in Section 2 of the paper.
%   [dyn_zero_to_t_plan, dyn_t_plan_to_t_total] = generate_parameterized_dynamics(t_plan, t_total)
%   starting from initial velocity k_i^v, we use constant acceleration over
%   t \in [0, t_plan]. 
%   then, we define trajectories with a failsafe
%   (braking) maneuver from the peak speed over t \in [t_plan, t_total].


currentFile = mfilename('fullpath');
gen_path = fileparts(currentFile);
save_path = fullfile(gen_path,'traj_model');

if ~exist(save_path, 'dir')
   mkdir(save_path)
end


syms p_x p_y kvx kax kvy kay t real;
syms udummy real; % CORA will require these arguments, but we won't use thems

x = [p_x; p_y; kvx; kvy; kax; kay; t];

% (1) we specify the dynamics in the first piece
dpx  = kvx + kax*t;
dpy  = kvy + kay*t;
dkvx = 0;
dkvy = 0;
dkax = 0;
dkay = 0;
dt   = 1;

dx = [dpx; dpy; dkvx; dkvy; dkax; dkay; dt];

dyn_zero_to_t_plan = matlabFunction(dx, 'File', fullfile(save_path,'traj_model_accel'), 'vars', {x, udummy});

% now we specify braking dynamics on t \in [t_plan, t_total]
t_to_stop = t_total - t_plan;
vx_dot_pk = kvx + kax * t_plan;
vy_dot_pk = kvy + kay * t_plan;
braking_acceleration_x = (0 - vx_dot_pk)/t_to_stop; % brake to 0 velocity from q_i_dot_pk in t_to_stop seconds
braking_acceleration_y = (0 - vy_dot_pk)/t_to_stop;

dpx = vx_dot_pk + braking_acceleration_x * (t-t_plan);
dpy = vy_dot_pk + braking_acceleration_y * (t-t_plan);
dx = [dpx; dpy; dkvx; dkvy; dkax; dkay; dt];
dyn_t_plan_to_t_total = matlabFunction(dx, 'File', fullfile(save_path,'traj_model_stop'), 'vars', {x, udummy});

end
