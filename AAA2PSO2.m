clc;
clear;
close all;
load A1.mat; % Load data from the file
N_acts = n; % Number of activities to process
% Configure particle swarm optimization (PSO) parameters
p.popsize = 1200; % Number of particles
p.dim = 6; % Dimensionality of the problem, i.e., the number of parameters to optimize
p.T = 2000; % Maximum number of iterations
% Set parameter bounds
lb = LB;  % Lower bound
ub = UB;  % Upper bound
% PSO parameters
w = W;  % Initial inertia weight
c1 = C1;  % Cognitive parameter
c2 = C2;  % Social parameter
% Early stopping parameters
patience = 120; % Patience parameter for early stopping, represents the maximum iterations without significant improvement
best_fitness_history = inf(patience, 1); % Record of the best fitness values over the last 'patience' iterations
global_fitness_history = []; % Record of the global best fitness values at each iteration
% Initialize particle positions and velocities
pos = rand(p.popsize, p.dim) .* (ub - lb) + lb; % Initialize particle positions randomly within the bounds
vel = zeros(p.popsize, p.dim); % Initialize velocities to zero
% Initialize personal best and global best
pbest = pos; % Each particle's personal best position starts as its initial position
pbest_fitness = inf(p.popsize, 1); % Initial personal best fitness is set to infinity
gbest = zeros(1, p.dim); % Global best position starts as zero
gbest_fitness = inf; % Initial global best fitness is set to infinity
for i = 1:N_acts
    target_val = A_vel(1:end, i); % Target fitting curve
    M_act = Mact(1:end, i); % Muscle activation
    Angle = A_ang(1:end, i); % Joint angle
    %% Define the objective function to minimize fitting error
    objective = @(s) sum((target_val - cumsum(((s(1) + s(2) * Angle + s(3) * Angle.^2) .* M_act + s(4) * exp(s(5) * Angle) - s(6) * sin(Angle)), 1)).^2);
    % PSO algorithm
    for t = 1:p.T
        % Dynamically adjust the inertia weight, decreasing with iterations
        w = 0.9 - (0.5 * t / p.T);
        for j = 1:p.popsize
            % Compute the fitness of the current particle
            fitness = objective(pos(j, :));
            % Update personal best position and fitness
            if fitness < pbest_fitness(j)
                pbest(j, :) = pos(j, :); % Update personal best position
                pbest_fitness(j) = fitness; % Update personal best fitness
            end
            % Update global best position and fitness
            if fitness < gbest_fitness
                gbest = pos(j, :); % Update global best position
                gbest_fitness = fitness; % Update global best fitness
            end
        end
        % Record the current global best fitness
        global_fitness_history(end + 1) = gbest_fitness;
        % Output the current iteration information
        fprintf('Iteration %d, Global Best Fitness: %f\n', t, gbest_fitness);
        % Update the early stopping history
        best_fitness_history(mod(t-1, patience) + 1) = gbest_fitness;
        % Check for early stopping condition
        if t > patience && all(abs(diff(best_fitness_history)) < 1e-7)
            fprintf('Early stopping at iteration %d\n', t);
            break;
        end
        % Update the velocity and position of the particles
        for j = 1:p.popsize
            vel(j, :) = w * vel(j, :) + c1 * rand * (pbest(j, :) - pos(j, :)) + c2 * rand * (gbest - pos(j, :));
            pos(j, :) = pos(j, :) + vel(j, :);
            % Ensure particles stay within the bounds
            pos(j, :) = max(min(pos(j, :), ub), lb);
        end
    end
    % Further optimize the global best position using fminunc
    options_fminunc = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxFunctionEvaluations', 1000);
    bestx = fminunc(objective, gbest, options_fminunc);
    %% Store the best parameters
    Bests(i, :) = bestx;
end
% Save the best parameters
save A2 Bests;
