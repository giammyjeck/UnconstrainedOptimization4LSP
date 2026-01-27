% function [outputArg1,outputArg2] = tuning(inputArg1,inputArg2)
% %UNTITLED Summary of this function goes here
% %   Detailed explanation goes here
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;
% end


%% =============================================
%% PARAMETER TUNING FOR MODIFIED NEWTON METHOD
%% Progressive grid search with early stopping
%% =============================================
clear; clc; close all;
addpath(genpath(pwd));
addpath(genpath("C:/Users/Utente/Desktop/Corsi/Numerical optimization for large scale problems and Stochastic Optimization/NumericalO4LSP/main"));



rng(346710);

%% ---------------- Problem definition ----------------
[f, gradf, hessf, xbar_gen] = problem_trig16();

%% ---------------- Fixed parameters ----------------
kmax    = 1000;      % Max outer iterations
tolgrad = 1e-6;      % Convergence criterion

%% ---- Choosing c1 carefully ----
% Must be small for Armijo but not of order eps/10 to avoid
% corrections comparable with machine precision
c1      = 1e-4; % adattare con rho

%% ---------------- Grid parameters ----------------
gamma_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];   % scaling for beta
rho_grid   = [0.3, 0.5, 0.8];                 % backtracking contraction
btmax_grid = [10, 20];                        % max backtracking iterations

alpha_loss = 0.01;  % weight for iterations in loss

%% ---------------- PHASE 1: COARSE SCREENING ----------------
fprintf('=== PHASE 1: COARSE SCREENING ===\n');
dims_screen = 1e3;    % cheap dimension
n_starts_1  = 5;      % few starting points
kmax_screen = 100;     % early stopping for coarse screening

candidates = [];       % list of configurations that pass screening
cfg_log = {};          % tracking info for all tested configs

cfg_id = 0;
for gamma = gamma_grid
    for rho = rho_grid
        for btmax = btmax_grid

            cfg_id = cfg_id + 1;
            beta = gamma * dims_screen;  % scaling law
            success = true;
            fail_reason = '';

            % Starting points
            x0_base = xbar_gen(dims_screen);
            x0_rand = (x0_base - 1) + 2*rand(dims_screen, n_starts_1-1);
            all_x0 = [x0_base, x0_rand];

            for s = 1:size(all_x0,2)
                [~, ~, gnorm, k] = modified_newton_method(...
                    all_x0(:,s), f, gradf, hessf, ...
                    kmax_screen, tolgrad, c1, rho, btmax, beta);

                % Early stop if fails
                if gnorm > tolgrad
                    success = false;
                    fail_reason = 'grad_norm_high';
                    break;
                end
                if k >= kmax_screen
                    success = false;
                    fail_reason = 'max_iter';
                    break;
                end
            end

            % Store config log
            cfg_log{cfg_id}.gamma = gamma;
            cfg_log{cfg_id}.rho   = rho;
            cfg_log{cfg_id}.btmax = btmax;
            cfg_log{cfg_id}.phase = 1;
            cfg_log{cfg_id}.success = success;
            cfg_log{cfg_id}.fail_reason = fail_reason;

            % Keep candidates that pass screening
            if success
                candidates = [candidates; gamma, rho, btmax];
                fprintf('  kept: gamma=%g rho=%g btmax=%d\n', gamma, rho, btmax);
            else
                fprintf('  discarded: gamma=%g rho=%g btmax=%d | reason: %s\n', ...
                    gamma, rho, btmax, fail_reason);
            end
    end
    end
end
fprintf('Candidates after screening: %d\n', size(candidates,1));

%% ---------------- PHASE 2: REFINEMENT ----------------

fprintf('\n=== PHASE 2: REFINEMENT ===\n');
dims_ref   = [1e3, 1e4, 1e5];  % two representative dimensions
n_starts_2 = 10;           % starting points per dimension

refined = struct();
r_id = 0;

for i = 1:size(candidates,1)
    gamma = candidates(i,1);
    rho   = candidates(i,2);
    btmax = candidates(i,3);

    total_time = 0;
    total_iters = 0;
    n_runs = 0;
    success = true;
    fail_reason = '';

    for n = dims_ref
        beta = gamma * n;

        % Generate starting points
        x0_base = xbar_gen(n);
        x0_rand = (x0_base - 1) + 2*rand(n, n_starts_2-1);
        all_x0  = [x0_base, x0_rand];

        for s = 1:size(all_x0,2)
            tic;
            [~, ~, gnorm, k] = modified_newton_method(...
                all_x0(:,s), f, gradf, hessf, ...
                kmax, tolgrad, c1, rho, btmax, beta);
            t = toc;

            % Early stop if fails
            if gnorm > tolgrad
                success = false;
                fail_reason = sprintf('grad_norm_high at n=%d, start=%d', n, s);
                break;
            end
            if k >= kmax
                success = false;
                fail_reason = sprintf('max_iter at n=%d, start=%d', n, s);
                break;
            end

            total_time  = total_time + t;
            total_iters = total_iters + k;
            n_runs = n_runs + 1;
        end

        if ~success, break; end
    end

    r_id = r_id + 1;
    refined(r_id).gamma = gamma;
    refined(r_id).rho   = rho;
    refined(r_id).btmax = btmax;
    refined(r_id).success = success;
    refined(r_id).fail_reason = fail_reason;

    if success
        % compute multi-dimensional loss
        avg_time  = total_time / n_runs;
        avg_iters = total_iters / n_runs;
        refined(r_id).avg_time  = avg_time;
        refined(r_id).avg_iters = avg_iters;
        refined(r_id).loss = avg_time + alpha_loss*avg_iters;

        fprintf('  kept: gamma=%g rho=%g btmax=%d | loss=%.4f\n', ...
            gamma, rho, btmax, refined(r_id).loss);
    else
        refined(r_id).loss = Inf;
        fprintf('  discarded: gamma=%g rho=%g btmax=%d | reason: %s\n', ...
            gamma, rho, btmax, fail_reason);
    end
end

%% ---------------- PHASE 3: FINAL SELECTION ----------------
fprintf('\n=== PHASE 3: FINAL SELECTION ===\n');

losses = [refined.loss];
[~, best_idx] = min(losses);
best = refined(best_idx);

fprintf('\nBEST CONFIGURATION FOUND:\n');
fprintf(' gamma  = %.1e\n', best.gamma);
fprintf(' rho    = %.2f\n', best.rho);
fprintf(' btmax  = %d\n', best.btmax);
fprintf(' avg time  = %.4f s\n', best.avg_time);
fprintf(' avg iters = %.2f\n', best.avg_iters);
fprintf(' loss      = %.4f\n', best.loss);

%% ---------------- PRINT SUMMARY OF REFINEMENT ----------------
fprintf('\n=== SUMMARY OF CONFIGURATIONS TESTED ===\n');
for i = 1:length(refined)
    if refined(i).success
        status = 'kept';
    else
        status = sprintf('discarded (%s)', refined(i).fail_reason);
    end
    fprintf('gamma=%.1e rho=%.2f btmax=%d -> %s\n', ...
        refined(i).gamma, refined(i).rho, refined(i).btmax, status);
end

%% ---------------- PRINT SUMMARY OF REFINEMENT WITH LOSS ----------------
fprintf('\n=== SUMMARY OF REFINEMENT CONFIGURATIONS WITH LOSS ===\n');
fprintf('%-10s %-8s %-6s %-10s %-10s %-10s\n', ...
    'gamma', 'rho', 'btmax', 'status', 'avg_time', 'loss');

for i = 1:length(refined)
    if refined(i).success
        status = 'kept';
        avg_time = refined(i).avg_time;
        loss_val = refined(i).loss;
    else
        status = sprintf('discarded (%s)', refined(i).fail_reason);
        avg_time = NaN;
        loss_val = Inf;
    end
    fprintf('%-10.1e %-8.2f %-6d %-10s %-10.4f %-10.4f\n', ...
        refined(i).gamma, refined(i).rho, refined(i).btmax, status, avg_time, loss_val);
end

% Show how the best is selected
fprintf('\n=== BEST CONFIGURATION SELECTED BASED ON LOSS ===\n');
fprintf('gamma = %.1e, rho = %.2f, btmax = %d\n', ...
    best.gamma, best.rho, best.btmax);
fprintf('avg_time = %.4f s, avg_iters = %.2f, loss = %.4f\n', ...
    best.avg_time, best.avg_iters, best.loss);
