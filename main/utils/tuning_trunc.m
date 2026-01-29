
clear; clc; close all;
addpath(genpath("C:\Users\Utente\Desktop\Corsi\Numerical optimization for large scale problems and Stochastic Optimization\NumericalO4LSP\main"));

rng(346710);

% ---------------- Problem ----------------
[f, gradf, hessf, xbar_gen] = problem14();

% ---------------- Experiment settings ----------------
tolgrad   = 1e-6;

% rho levels to explore (user requested 3 livelli)
rho_levels = [0.3,0.5,0.8];

% baseline initial values
c1_baseline   = 1e-4;
kmax_baseline = 10;
bt_baseline   = 1;
max_cg_baseline = 5;

% escalation levels (kept aside, attivati on-demand)
c1_levels   = [1e-4, 1e-3];
kmax_levels = [10, 50, 100, 200, 2000, 5000];
bt_levels   = [1, 5, 10, 20, 30];
max_cg_levels = [5, 10, 20, 40, 100, 500, 1000];

% runtime sampling for refinement stages
n_ref1 = 1e3;
n_starts_ref1 = 3;   % cheap screening

n_ref2 = [1e3, 1e4];
n_starts_ref2 = 5;  % robust evaluation

% loss weights
w_t = 1;
w_g = 10;

% containers
good_configs = [];   % configs that pass ref1
tested_keys = containers.Map('KeyType','char','ValueType','logical');

% ---------------- Queue initialization ----------------
queue = {}; qhead = 1;
for r = rho_levels
    cfg.rho = r;
    cfg.c1 = c1_baseline;
    cfg.kmax = kmax_baseline;
    cfg.bt   = bt_baseline;
    cfg.max_cg = max_cg_baseline;
    cfg.origin = 'init';
    queue{end+1} = cfg;
end

fprintf('Initial queue length: %d\n', numel(queue));

% helper key (no beta)
make_key = @(cfg) sprintf('r%.6g_c%.6g_k%d_bt%d_cg%d', cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);

% ---------------- REFINEMENT 1: adaptive queue processing ----------------
fprintf('\n=== REFINEMENT 1: adaptive queue processing ===\n');
while qhead <= numel(queue)
    cfg = queue{qhead}; qhead = qhead + 1;    % pop front
    key = make_key(cfg);
    if tested_keys.isKey(key)
        fprintf('skip (tested): rho=%g c1=%.0e kmax=%d bt=%d max_cg=%d\n', ...
            cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);
        continue;
    end

    fprintf('\nTesting config: rho=%g c1=%.0e kmax=%d bt=%d max_cg=%d\n', ...
        cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);

    % generate starting points
    x0_base = xbar_gen(n_ref1);
    x0_rand = (x0_base - 1) + 2*rand(n_ref1, n_starts_ref1-1);
    all_x0 = [x0_base, x0_rand];

    % flags
    runs_success = true;
    any_bt_reached = false;
    any_k_reached = false;
    any_cg_reached = false;
    structural_fail = false;

    % test per start
    for s = 1:size(all_x0,2)
        x0 = all_x0(:,s);
        
        % truncated_newton_method signature:
        % [xk,fk,gradfk_norm,k,xseq,btseq,pks,inner_iters] = ...
        [~, ~, gradnorm, k, ~, btseq, ~, inner_iters] = truncated_newton_method( ...
            x0, f, gradf, hessf, cfg.kmax, tolgrad, cfg.c1, cfg.rho, cfg.bt, cfg.max_cg);

        bt_reached = (~isempty(btseq) && btseq(end) >= cfg.bt);
        k_reached  = (k >= cfg.kmax);
        cg_reached = (~isempty(inner_iters) && inner_iters(end) >= cfg.max_cg);

        if bt_reached 
            any_bt_reached = true;
            runs_success = false;
        end   
        if cg_reached 
            any_cg_reached = true;
            runs_success = false;
        end
        if k_reached 
            any_k_reached = true;
            runs_success = false;
        end
        if gradnorm > tolgrad
            runs_success = false;
        end

    end

    % mark tested
    tested_keys(key) = true;

    if runs_success
        cfg.note = 'passed_ref1';
        good_configs = [good_configs; cfg]; %#ok<AGROW>
        fprintf('  -> accepted (passed all %d starts)\n', size(all_x0,2));
        continue;
    end

    % ESCALATION LOGIC
    % Priority: bt (+c1 fallback) -> max_cg -> kmax -> discard

    if any_bt_reached
        idx = find(bt_levels > cfg.bt, 1, 'first');
        if ~isempty(idx)
            newcfg = cfg; newcfg.bt = bt_levels(idx);
            newkey = make_key(newcfg);
            if ~tested_keys.isKey(newkey)
                queue{end+1} = newcfg;
                fprintf('  -> enqueued bt escalation: new bt=%d\n', newcfg.bt);
            end
        end
        if cfg.c1 == c1_baseline
            newcfg2 = cfg; newcfg2.c1 = c1_levels(end);
            newkey2 = make_key(newcfg2);
            if ~tested_keys.isKey(newkey2)
                queue{end+1} = newcfg2;
                fprintf('  -> enqueued c1 fallback: c1=%.0e\n', newcfg2.c1);
            end
        end
        continue;
    end

    if any_cg_reached
        idx_cg = find(max_cg_levels > cfg.max_cg, 1, 'first');
        if ~isempty(idx_cg)
            newcfg = cfg; newcfg.max_cg = max_cg_levels(idx_cg);
            newkey = make_key(newcfg);
            if ~tested_keys.isKey(newkey)
                queue{end+1} = newcfg;
                fprintf('  -> enqueued max_cg escalation: new max_cg=%d\n', newcfg.max_cg);
            end
        else
            fprintf('  -> max_cg saturated (no larger level), skipping\n');
        end
        continue;
    end

    if any_k_reached
        idxk = find(kmax_levels > cfg.kmax, 1, 'first');
        if ~isempty(idxk)
            newcfg = cfg; newcfg.kmax = kmax_levels(idxk);
            newkey = make_key(newcfg);
            if ~tested_keys.isKey(newkey)
                queue{end+1} = newcfg;
                fprintf('  -> enqueued kmax escalation: new kmax=%d\n', newcfg.kmax);
            end
        else
            fprintf('  -> kmax saturated (no larger level), skipping\n');
        end
        continue;
    end

    fprintf('  -> fallback: discarded (no escalation rule applicable)\n');
end

fprintf('\nRefinement 1 complete. Good configs found: %d\n', numel(good_configs));

% deduplicate good_configs
if ~isempty(good_configs)
    keys_good = arrayfun(@(c) make_key(c), good_configs, 'UniformOutput', false);
    [~, ia] = unique(keys_good, 'stable');
    good_configs = good_configs(ia);
    fprintf('After deduplication good configs: %d\n', numel(good_configs));
end

% print tested configs
fprintf('\n--- Tested configurations (summary) ---\n');
tk = keys(tested_keys);
for i = 1:numel(tk)
    fprintf('%s\n', tk{i});
end

% ---------------- REFINEMENT 2: robust evaluation and loss ----------------
fprintf('\n=== REFINEMENT 2: robust evaluation ===\n');
refined = [];
for i = 1:numel(good_configs)
    cfg = good_configs(i);
    L_cfg = -Inf;
    success_cfg = true;
    fprintf('\nEvaluating config #%d: rho=%g c1=%.0e kmax=%d bt=%d max_cg=%d\n', ...
        i, cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);

    for n = n_ref2
        x0_base = xbar_gen(n);
        x0_rand = (x0_base - 1) + 2*rand(n, n_starts_ref2-1);
        all_x0 = [x0_base, x0_rand];

        for s = 1:size(all_x0,2)
            x0 = all_x0(:,s);
            
            tic;
            [~, ~, gradnorm, k, ~, btseq, ~, inner_iters] = truncated_newton_method( ...
                x0, f, gradf, hessf, cfg.kmax, tolgrad, cfg.c1, cfg.rho, cfg.bt, cfg.max_cg);
            t_run = toc;


            if gradnorm > tolgrad && ( k >= cfg.kmax || (~isempty(btseq) && btseq(end) >= cfg.bt) || ...
               (~isempty(inner_iters) && inner_iters(end) >= cfg.max_cg))
                fprintf('  run failed at n=%d start=%d: gradnorm=%.2e k=%d bt_last=%d cg_last=%d\n', ...
                    n, s, gradnorm, k, (~isempty(btseq) * btseq(end)), (~isempty(inner_iters) * inner_iters(end)));
                success_cfg = false;
                break;
            end

            phi = max(0, log10(gradnorm / tolgrad));
            L_run = w_t * t_run + w_g * phi;
            L_cfg = max(L_cfg, L_run);
        end
        if ~success_cfg, break; end
    end

    if success_cfg
        r = struct();
        r.max_cg = cfg.max_cg;
        r.rho  = cfg.rho;
        r.c1   = cfg.c1;
        r.kmax = cfg.kmax;
        r.bt   = cfg.bt;
        r.loss = L_cfg;
        refined = [refined; r]; %#ok<AGROW>
        fprintf('  kept in REF2: loss=%.4e\n', L_cfg);
    else
        fprintf('  discarded in REF2 (not robust)\n');
    end
end

if isempty(refined)
    error('No configuration survived refinement 2.');
end

% ---------------- STEP 5: select top 10% and plot ----------------
losses = [refined.loss];
pct = 10;
thr = prctile(losses, pct);
best_idx = find(losses <= thr);
best = refined(best_idx);

fprintf('\n=== FINAL SELECTION: top %d%% (loss <= %.4g) ===\n', pct, thr);
for i = 1:numel(best)
    fprintf('%d) rho=%.6g c1=%.0e kmax=%d bt=%d max_cg=%d loss=%.4e\n', ...
        i, best(i).rho, best(i).c1, best(i).kmax, best(i).bt, best(i).max_cg, best(i).loss);
end

% plots
rho_vals   = [refined.rho];
c1_vals    = [refined.c1];
bt_vals    = [refined.bt];
cg_vals    = [refined.max_cg];
loss_vals  = [refined.loss];

figure; scatter(rho_vals, c1_vals, 80, log10(loss_vals), 'filled'); colorbar;
xlabel('\rho'); ylabel('c1'); title('All refined configs: color = log_{10}(loss)'); grid on;

figure; scatter(bt_vals, cg_vals, 80, log10(loss_vals), 'filled'); colorbar;
xlabel('bt'); ylabel('max\_cg'); title('bt vs max\_cg (log_{10} loss)'); grid on;

figure; scatter(rho_vals, cg_vals, 80, log10(loss_vals), 'filled'); colorbar;
xlabel('\rho'); ylabel('max\_cg'); title('\rho vs max\_cg (log_{10} loss)'); grid on;

% highlight best on first plot
hold on;
for i = 1:numel(best)
    scatter(best(i).rho, best(i).c1, 150, 'k', 'LineWidth', 1.5);
end
hold off;

fprintf('\nScript finished. Summary: initial structural configs=%d, good after ref1=%d, refined after ref2=%d, final selected=%d\n', ...
    numel(rho_levels), numel(good_configs), numel(refined), numel(best));
