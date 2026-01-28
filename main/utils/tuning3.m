
%% ============================================================
%% PARAMETER TUNING FOR MODIFIED NEWTON METHOD
%% Adaptive kmax + conditional c1 fallback
%% ============================================================
clear; clc; close all;
addpath(genpath(pwd));

rng(346710);

% Problem
[f, gradf, hessf, xbar_gen] = problem_broyden31();

%% ---------------- Parameters ----------------
tolgrad   = 1e-6;        % experimenter choice (not tuned)
c1_small  = 1e-4;        % tried first
c1_large  = 1e-3;        % tried only if btmax+near failure

% Dimensions for refinements
n_ref1 = 1e3;
n_ref2 = [1e3, 1e4];

% Beta grid (we tune beta, not gamma)
beta_grid = [1e-3, 1e-2, 1e-1, 1];

% Backtracking rho grid
rho_grid = [0.3, 0.5, 0.8];

% kmax candidates (will be tried adaptively, ascending)
kmax_grid = [500, 1000, 2000];

% btmax selection control (rho-dependent)
alpha_min    = 1e-8;   % minimal admissible step: rho^btmax >= alpha_min
btmax_min    = 3;
n_bt_samples = 4;

% diagnostic threshold for "near stationarity" when btmax reached
kappa_grad = 10;  % gnorm <= kappa_grad * tolgrad => considered near

% runtime sampling
n_starts_ref1 = 5;   % cheap
n_starts_ref2 = 10;  % more thorough

% Loss weights (used in ref2 / final)
w_t = 1;
w_g = 10;

%% ---------------- Generate structural configurations (no kmax) ----------------
cfg_id = 0;
configs = struct();

for beta = beta_grid
    for rho = rho_grid
        if rho <= 0 || rho >= 1
            continue;
        end
        btmax_bound = floor(log(alpha_min) / log(rho));
        if btmax_bound < btmax_min
            continue;
        end
        if btmax_bound == btmax_min
            btmax_grid_rho = btmax_min;
        else
            btmax_grid_rho = unique(round( ...
                linspace(btmax_min, btmax_bound, ...
                min(n_bt_samples, btmax_bound - btmax_min + 1))));
        end

        for btmax = btmax_grid_rho
            cfg_id = cfg_id + 1;
            configs(cfg_id).beta     = beta;
            configs(cfg_id).rho      = rho;
            configs(cfg_id).btmax    = btmax;
            configs(cfg_id).alive    = true;   % alive after generation
            configs(cfg_id).c1_used  = NaN;    % to be set if config survives
            configs(cfg_id).kmax_used= NaN;    % minimal kmax that succeeded
        end
    end
end

fprintf('Total structural configurations generated: %d\n', numel(configs));

%% ---------------- REFINEMENT 1 (adaptive kmax, conditional c1) ----------------
fprintf('\n=== REFINEMENT 1 (n = %d) ===\n', n_ref1);
dead_ref1 = 0;

for i = 1:numel(configs)
    if ~configs(i).alive
        continue;
    end

    beta  = configs(i).beta;
    rho   = configs(i).rho;
    btmax = configs(i).btmax;

    % generate starting points (cheap)
    x0_base = xbar_gen(n_ref1);
    x0_rand = (x0_base - 1) + 2*rand(n_ref1, n_starts_ref1-1);
    all_x0  = [x0_base, x0_rand];

    config_survived = false;
    recorded_kmax = NaN;
    recorded_c1 = NaN;

    % try kmax candidates in ascending order
    for kmax = kmax_grid

        % flags per this kmax
        need_increase_kmax = false;
        need_try_c1_large = false;
        structural_fail = false;  % grad_not_converged or bt_far

        % first attempt with c1_small
        c1_try = c1_small;

        for s = 1:size(all_x0,2)
            [~, ~, gnorm, k, ~, btseq] = modified_newton_method( ...
                all_x0(:,s), f, gradf, hessf, ...
                kmax, tolgrad, c1_try, rho, btmax, beta);

            bt_reached = (~isempty(btseq) && btseq(end) >= btmax);

            if gnorm > tolgrad
                % not converged to tol
                if bt_reached
                    % distinguish near vs far
                    if gnorm <= kappa_grad * tolgrad
                        % near -> relax c1 may help
                        need_try_c1_large = true;
                    else
                        % far -> structural fail (H/dir problem)
                        structural_fail = true;
                        break;
                    end
                else
                    % failed not due to btmax (likely direction problem)
                    structural_fail = true;
                    break;
                end
            elseif k >= kmax
                % reached kmax -> need larger budget
                need_increase_kmax = true;
                % keep checking other starts to confirm if all do
            else
                % this start succeeded for this kmax,c1
                % nothing to do
            end
        end % loop starts

        if structural_fail
            % discard entire configuration (for all kmax)
            configs(i).alive = false;
            dead_ref1 = dead_ref1 + 1;
            fprintf(' discarded (structural): beta=%g rho=%g btmax=%d | cause=structural\n', ...
                beta, rho, btmax);
            break; % break kmax loop -> next config
        end

        if need_try_c1_large
            % try same kmax but with c1_large
            c1_try2 = c1_large;
            structural_fail2 = false;
            need_increase_kmax2 = false;
            for s = 1:size(all_x0,2)
                [~, ~, gnorm2, k2, ~, btseq2] = modified_newton_method( ...
                    all_x0(:,s), f, gradf, hessf, ...
                    kmax, tolgrad, c1_try2, rho, btmax, beta);

                bt_reached2 = (~isempty(btseq2) && btseq2(end) >= btmax);

                if gnorm2 > tolgrad
                    if bt_reached2
                        if gnorm2 <= kappa_grad * tolgrad
                            % still near even with larger c1 -> try increase kmax
                            need_increase_kmax2 = true;
                        else
                            structural_fail2 = true;
                            break;
                        end
                    else
                        structural_fail2 = true;
                        break;
                    end
                elseif k2 >= kmax
                    need_increase_kmax2 = true;
                else
                    % success for this start with c1_large
                end
            end

            if structural_fail2
                configs(i).alive = false;
                dead_ref1 = dead_ref1 + 1;
                fprintf(' discarded (structural even with c1_large): beta=%g rho=%g btmax=%d\n', ...
                    beta, rho, btmax);
                break; % next config
            end

            if ~need_increase_kmax2
                % succeeded with c1_large at this kmax
                config_survived = true;
                recorded_kmax = kmax;
                recorded_c1  = c1_try2;
                break; % stop kmax loop -> config survived
            else
                % need to increase kmax (after trying c1_large)
                % continue to next kmax
                continue;
            end

        end % end try c1_large

        if ~need_increase_kmax && ~need_try_c1_large
            % success with c1_small and this kmax
            config_survived = true;
            recorded_kmax = kmax;
            recorded_c1  = c1_try;
            break;
        end

        if need_increase_kmax
            % try next larger kmax
            continue;
        end

    end % end for kmax

    if config_survived
        configs(i).c1_used   = recorded_c1;
        configs(i).kmax_used = recorded_kmax;
        fprintf(' kept: beta=%g rho=%g btmax=%d | kmax_used=%d c1=%.0e\n', ...
            beta, rho, btmax, recorded_kmax, recorded_c1);
        % NOTE: do NOT prune siblings on success (avoid bias)
    else
        if configs(i).alive  % it means we exited kmax loop without success and without structural fail -> final discard
            configs(i).alive = false;
            dead_ref1 = dead_ref1 + 1;
            fprintf(' discarded (insufficient budget): beta=%g rho=%g btmax=%d | tried all kmax\n', ...
                beta, rho, btmax);
        end
    end
end

fprintf('Refinement 1: discarded configurations: %d / %d\n', dead_ref1, numel(configs));

%% ---------------- REFINEMENT 2 (robustness across n_ref2) ----------------
fprintf('\n=== REFINEMENT 2 (n = %g, %g) ===\n', n_ref2(1), n_ref2(2));
refined = struct();
rid = 0;
dead_ref2 = 0;

for i = 1:numel(configs)
    if ~configs(i).alive
        continue;
    end

    beta  = configs(i).beta;
    rho   = configs(i).rho;
    btmax = configs(i).btmax;
    c1    = configs(i).c1_used;
    kmax_used = configs(i).kmax_used;
    if isnan(c1)
        c1 = c1_small; % fallback if unset
    end
    if isnan(kmax_used)
        kmax_try_list = kmax_grid; % try max budgets
    else
        % start from discovered kmax_used but allow escalation if needed
        kmax_try_list = kmax_grid(kmax_grid >= kmax_used);
        if isempty(kmax_try_list)
            kmax_try_list = kmax_grid(end);
        end
    end

    success_cfg = true;
    L_cfg = -Inf;

    % For each dimension
    for n = n_ref2
        gamma = beta / n;

        % generate starts
        x0_base = xbar_gen(n);
        x0_rand = (x0_base - 1) + 2*rand(n, n_starts_ref2-1);
        all_x0  = [x0_base, x0_rand];

        % attempt with adaptive kmax list for this config
        success_for_n = false;
        for ktry = kmax_try_list
            need_increase = false;
            structural_fail = false;

            for s = 1:size(all_x0,2)
                tic;
                [~, ~, gnorm, k, ~, btseq] = modified_newton_method( ...
                    all_x0(:,s), f, gradf, hessf, ...
                    ktry, tolgrad, c1, rho, btmax, gamma*n);
                t_run = toc;

                bt_reached = (~isempty(btseq) && btseq(end) >= btmax);

                if gnorm > tolgrad
                    if bt_reached
                        % if near, we might try larger k; but here we treat as fail
                        structural_fail = true;
                        break;
                    end
                elseif k >= ktry
                    need_increase = true;
                    break;
                else
                    % a successful start: accumulate stats
                end
            end % starts

            if structural_fail
                % this config fails on this dimension
                success_for_n = false;
                break; % no need to try larger k for this dimension
            end

            if need_increase
                % try next larger ktry
                continue;
            else
                % succeeded for all starts on this dimension with ktry
                success_for_n = true;
                break;
            end
        end % ktry

        if ~success_for_n
            success_cfg = false;
            dead_ref2 = dead_ref2 + 1;
            fprintf(' discarded in REF2: beta=%g rho=%g btmax=%d | failed on n=%d\n', ...
                beta, rho, btmax, n);
            break; % break n loop
        else
            % compute loss on this dimension using one run per start (aggregate max)
            % (for speed use only first start to compute t and gnorm more precisely if desired)
            % Here we measure one representative run to form L_run (you can extend to average)
            tic;
            [~, ~, gnorm_r, k_r, ~, btseq_r] = modified_newton_method( ...
                all_x0(:,1), f, gradf, hessf, ...
                ktry, tolgrad, c1, rho, btmax, gamma*n);
            t_run = toc;
            phi = max(0, log10(gnorm_r / tolgrad));
            L_run = w_t * t_run + w_g * phi;
            L_cfg = max(L_cfg, L_run);
        end
    end % dimensions

    if success_cfg
        rid = rid + 1;
        refined(rid).beta   = beta;
        refined(rid).rho    = rho;
        refined(rid).btmax  = btmax;
        refined(rid).kmax   = configs(i).kmax_used;
        refined(rid).c1     = c1;
        refined(rid).loss   = L_cfg;
        fprintf(' kept REF2: beta=%g rho=%g btmax=%d | loss=%.3e (c1=%.0e)\n', ...
            beta, rho, btmax, L_cfg, c1);
    end

end

fprintf('Refinement 2: discarded configurations: %d\n', dead_ref2);

%% ---------------- FINAL SELECTION & PLOTS ----------------
if isempty(refined)
    error('No configuration survived refinement 2.');
end

losses = [refined.loss];
thr = prctile(losses, 10);
best = refined(losses <= thr);

beta_vals  = [best.beta];
rho_vals   = [best.rho];
btmax_vals = [best.btmax];
kmax_vals  = [best.kmax];
c1_vals    = [best.c1];
loss_vals  = [best.loss];

% rho vs btmax
figure; scatter(btmax_vals, rho_vals, 80, log10(loss_vals), 'filled'); colorbar;
xlabel('btmax'); ylabel('\rho'); title('Best configs: \rho vs btmax (log_{10} loss)'); grid on;

% beta vs btmax
figure; scatter(btmax_vals, beta_vals, 80, log10(loss_vals), 'filled'); colorbar;
xlabel('btmax'); ylabel('\beta'); title('Best configs: \beta vs btmax (log_{10} loss)'); grid on;

% beta vs rho
figure; scatter(rho_vals, beta_vals, 80, log10(loss_vals), 'filled'); colorbar;
xlabel('\rho'); ylabel('\beta'); title('Best configs: \beta vs \rho (log_{10} loss)'); grid on;

% c1 vs btmax
figure; scatter(btmax_vals, c1_vals, 80, log10(loss_vals), 'filled'); colorbar;
xlabel('btmax'); ylabel('c1'); title('Best configs: c1 vs btmax (log_{10} loss)'); grid on;

% Top configurations
[~, idx_best] = sort(loss_vals);
topN = min(5, numel(best));
topk = best(idx_best(1:topN));

disp('Top configurations (lowest loss):');
for t = 1:topN
    fprintf('Beta=%.3g, rho=%.2g, btmax=%d, kmax=%d, c1=%.0e, loss=%.4e\n', ...
        topk(t).beta, topk(t).rho, topk(t).btmax, topk(t).kmax, topk(t).c1, topk(t).loss);
end

fprintf('\nSummary:\n  total structural configs: %d\n  discarded ref1: %d\n  discarded ref2: %d\n  survived to final: %d\n', ...
    numel(configs), dead_ref1, dead_ref2, numel(best));
