clc;
clear;
close all;
load A1.mat;
N_acts = n;
% Configure particle swarm optimization (PSO) parameters
p.popsize = 1500; % Number of particles
p.dim = 6; % Number of dimensions
p.T = 2000; % Maximum number of iterations
% Set the parameter boundaries
lb = LB;  % Lower bound
ub = UB;  % Upper bound
% PSO parameters
w = W;  % Initial inertia weight
c1 = C1;  % Cognitive parameter
c2 = C2;  % Social parameter
% Early stopping parameters
patience = 120; % Number of iterations to check for early stopping
best_fitness_history = inf(patience, 1); % History of the best fitness values
global_fitness_history = []; % To record the loss values
% Initialize position and velocity of particles
pos = rand(p.popsize, p.dim) .* (ub - lb) + lb; % Random initialization of particle positions within bounds
vel = zeros(p.popsize, p.dim); % Initialize velocity to zero
% Initialize personal best and global best
pbest = pos; % Personal best positions start as the initial positions
pbest_fitness = inf(p.popsize, 1); % Initialize personal best fitness to infinity
gbest = zeros(1, p.dim); % Global best position starts as zero
gbest_fitness = inf; % Initialize global best fitness to infinity
for i = 1:N_acts
    target_val = A_ang(1:end, i); % Target fitting curve
    M_act = Mact(1:end, i);       % Muscle activation
    Angle = A_ang(1:end, i);      % Joint angle
    %% Define the objective function to minimize the fitting error
    objective = @(s) sum((target_val - cumsum(cumsum(((s(1) + s(2) * Angle + s(3) * Angle.^2) .* M_act + s(4) * exp(s(5) * Angle) - s(6) * sin(Angle)), 1),1)).^2);
    % PSO algorithm
    for t = 1:p.T
        % Dynamically adjust the inertia weight
        w = 0.9 - (0.5 * t / p.T);
        for j = 1:p.popsize
            % Compute the fitness
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
        % Record the global best fitness value
        global_fitness_history(end + 1) = gbest_fitness;
        % Output the current iteration info
        fprintf('Iteration %d, Global Best Fitness: %f\n', t, gbest_fitness);
        % Update early stopping history
        best_fitness_history(mod(t-1, patience) + 1) = gbest_fitness;
        % Check for early stopping condition
        if t > patience && all(abs(diff(best_fitness_history)) < 1e-6)
            fprintf('Early stopping at iteration %d\n', t);
            break;
        end
        % Update velocity and position
        for j = 1:p.popsize
            vel(j, :) = w * vel(j, :) + c1 * rand * (pbest(j, :) - pos(j, :)) + c2 * rand * (gbest - pos(j, :));
            pos(j, :) = pos(j, :) + vel(j, :);
            % Ensure particles stay within bounds
            pos(j, :) = max(min(pos(j, :), ub), lb);
        end
    end
    % Further optimize using fminunc
    options_fminunc = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxFunctionEvaluations', 1000);
    bestx = fminunc(objective, gbest, options_fminunc);
    %% Store the best parameters
    Bests(i, :) = bestx;
end
% Save the results
save A2 Bests;
