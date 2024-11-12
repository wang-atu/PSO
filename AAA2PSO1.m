clc;
clear;
close all;
load A1.mat;
N_acts = n;  % Number of activities
% Configure particle swarm optimization (PSO) parameters
p.popsize = 1000;  % Number of particles
p.dim = 6;  % Dimensionality of the problem
p.T = 2000;  % Maximum number of iterations
% Set the parameter bounds
lb = LB;  % Lower bound
ub = UB;  % Upper bound
% PSO parameters
w = W;  % Initial inertia weight
c1 = C1;  % Cognitive parameter
c2 = C2;  % Social parameter
% Early stopping parameters
patience = 100;
best_fitness_history = inf(patience, 1);  % History of the best fitness values for early stopping
global_fitness_history = [];  % To record loss values
% Initial positions and velocities
pos = rand(p.popsize, p.dim) .* (ub - lb) + lb;  % Particle positions within the bounds
vel = zeros(p.popsize, p.dim);  % Initial velocities (zero)
% Personal best and global best
pbest = pos;  % Personal best positions
pbest_fitness = inf(p.popsize, 1);  % Personal best fitness values
gbest = zeros(1, p.dim);  % Global best position
gbest_fitness = inf;  % Global best fitness value
for i = 1:N_acts
    target_val = A_acc(1:end, i);  % Target fitting curve
    M_act = Mact(1:end, i);  % Muscle activation
    Angle = A_ang(1:end, i);  % Joint angle
    %% Define the objective function to minimize fitting error
    objective = @(s) sum((target_val - (((s(1) + s(2) * Angle + s(3) * Angle.^2) .* M_act + s(4) * exp(s(5) * Angle) - s(6) * sin(Angle)))).^2);
    % PSO algorithm
    for t = 1:p.T
        % Dynamically adjust inertia weight
        w = 0.9 - (0.5 * t / p.T);
        for j = 1:p.popsize
            % Compute fitness
            fitness = objective(pos(j, :));
            % Update personal best
            if fitness < pbest_fitness(j)
                pbest(j, :) = pos(j, :);
                pbest_fitness(j) = fitness;
            end
            % Update global best
            if fitness < gbest_fitness
                gbest = pos(j, :);
                gbest_fitness = fitness;
            end
        end
        % Record global best fitness
        global_fitness_history(end + 1) = gbest_fitness;
        % Real-time output
        fprintf('Iteration %d, Global Best Fitness: %f\n', t, gbest_fitness);
        % Update early stopping history
        best_fitness_history(mod(t-1, patience) + 1) = gbest_fitness;
        % Check if early stopping condition is met
        if t > patience && all(abs(diff(best_fitness_history)) < 1e-8)
            fprintf('Early stopping at iteration %d\n', t);
            break;
        end
        % Update velocity and position
        for j = 1:p.popsize
            vel(j, :) = w * vel(j, :) + c1 * rand * (pbest(j, :) - pos(j, :)) + c2 * rand * (gbest - pos(j, :));
            pos(j, :) = pos(j, :) + vel(j, :);
            % Ensure particles stay within the bounds
            pos(j, :) = max(min(pos(j, :), ub), lb);
        end
    end
    % Further optimization using fminunc
    options_fminunc = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxFunctionEvaluations', 1000);
    bestx = fminunc(objective, gbest, options_fminunc);
    %% Store the best parameters
    Bests(i, :) = bestx;
end
% Save results
save A2 Bests;
