clear; clc; close all;
 
% Aggiunge questa cartella e tutte le sottocartelle al path
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));
 
seed = 346710;
rng(seed);
 
% ---------------- Problem ----------------
%[f, grad_exact, hess_exact, xbar_gen, rfun, ~] = problem_broyden31();
[f, grad_exact, hess_exact, xbar_gen, xstarfun] = problem_trig16();
 
% ---------------- Modalita' di derivazione ----------------
% "exact" -> gradiente e Hessiana esatti
% "case1" -> gradiente esatto, Hessiana approssimata con differenze finite
% "case2" -> gradiente e Hessiana entrambi approssimati con differenze finite
deriv_mode = "case2";
k_fd    = 4;   % passo FD: h = 10^-k_fd
type_fd = 1;    % 1 = passo costante (h), 2 = passo relativo (hi)
 
switch deriv_mode
    case "exact"
        gradf = grad_exact;
        hessf = hess_exact;
    case "case1"
        gradf = grad_exact;
        %hessf = @(x) hess_fd_broyden31(grad_exact, x, k_fd, type_fd);
        hessf = @(x) trig_hess_fd_case1(grad_exact, x, k_fd, type_fd);
    case "case2"
        %gradf = @(x) grad_fd_broyden31(x, k_fd, type_fd, rfun);
        %hessf = @(x) hess_grad_fd_broyden31(x, k_fd, type_fd, rfun);
        gradf = @(x) trig_fd_case2_grad_only(x, k_fd, type_fd);
        hessf = @(x) trig_fd_case2_hess_only(x, k_fd, type_fd);
    otherwise
        error('deriv_mode non riconosciuto: %s', deriv_mode);
end
 
fprintf('Tuning truncated_newton_method - deriv_mode = %s', deriv_mode);
if deriv_mode ~= "exact"
    fprintf(' (k=%d, type=%d)', k_fd, type_fd);
end
fprintf('\n');
 
% ---------------- Parametri dell'esperimento ----------------
tolgrad = 1e-6;
 
% Parametro da esplorare nella griglia iniziale
rho_levels = [0.3, 0.5, 0.8];
 
% Valori di base per i parametri non esplorati inizialmente
c1_baseline     = 1e-4;
kmax_baseline   = 10;
bt_baseline     = 5;
max_cg_baseline = 2;
 
% Livelli di escalation, usati solo se la configurazione base fallisce
c1_levels     = [1e-4, 1e-3];
kmax_levels   = [10, 50, 100, 200,500,1000];
bt_levels     = [10, 20, 30, 40];
max_cg_levels = [5, 10, 1e2, 1e3, 1e4];
 
% Dimensioni usate nelle due fasi di valutazione
n_ref1         = 1e4;
n_starts_ref1  = 5;
 
n_ref2         = [2,1e3,1e4];
n_starts_ref2  = 5;
 
% Pesi della loss finale (tempo vs precisione raggiunta)
w_t = 1;
w_g = 10;
 
good_configs = [];
tested_keys  = containers.Map('KeyType', 'char', 'ValueType', 'logical');
 
% Registro dell'albero di ricerca (per il plot ad albero delle
% configurazioni esplorate durante il Refinement 1).
node_registry = containers.Map('KeyType', 'char', 'ValueType', 'any');
 
make_key = @(cfg) sprintf('r%.6g_c%.6g_k%d_bt%d_cg%d', ...
    cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);
 
% ---------------- Inizializzazione coda (Refinement 1) ----------------
queue = {};
for r = rho_levels
    cfg.rho    = r;
    cfg.c1     = c1_baseline;
    cfg.kmax   = kmax_baseline;
    cfg.bt     = bt_baseline;
    cfg.max_cg = max_cg_baseline;
    cfg.origin = 'init';
    key0 = make_key(cfg);
    register_node(node_registry, key0, '', 'griglia iniziale', cfg, 'queued');
    queue{end+1} = cfg; %#ok<SAGROW>
end
qhead = 1;
 
fprintf('Initial queue length: %d\n', numel(queue));
 
% ---------------- REFINEMENT 1: screening adattivo ----------------
fprintf('\n=== REFINEMENT 1: adaptive queue processing ===\n');
while qhead <= numel(queue)
    cfg = queue{qhead};
    qhead = qhead + 1;
 
    key = make_key(cfg);
    if tested_keys.isKey(key)
        fprintf('skip (gia'' testata): rho=%g c1=%.0e kmax=%d bt=%d max_cg=%d\n', ...
            cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);
        continue;
    end
 
    fprintf('\nTesting config: rho=%g c1=%.0e kmax=%d bt=%d max_cg=%d\n', ...
        cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);
 
    x0_base = xbar_gen(n_ref1);
    x0_rand = (x0_base - 1) + 2*rand(n_ref1, n_starts_ref1 - 1);
    all_x0  = [x0_base, x0_rand];
 
    runs_success    = true;
    any_bt_reached  = false;
    any_k_reached   = false;
    any_cg_reached  = false;
 
    for s = 1:size(all_x0, 2)
        [~, ~, gradnorm, k, ~, btseq, ~, inner_iters] = truncated_newton_method( ...
            all_x0(:,s), f, gradf, hessf, ...
            cfg.kmax, tolgrad, cfg.c1, cfg.rho, cfg.bt, cfg.max_cg);
 
        bt_reached = (~isempty(btseq) && any(btseq >= cfg.bt));
        k_reached  = (k >= cfg.kmax);
        cg_reached = (~isempty(inner_iters) && any(inner_iters >= cfg.max_cg));
        converged  = (gradnorm < tolgrad);
        if bt_reached
            any_bt_reached = true;
        end
        if cg_reached
            any_cg_reached = true;
        end
        if k_reached
            any_k_reached = true;
        end
        if bt_reached || cg_reached || k_reached || ~converged
            runs_success = false;
        end
    end
 
    tested_keys(key) = true;
 
    if runs_success
        cfg.note = 'passed_ref1';
        good_configs = [good_configs; cfg]; %#ok<AGROW>
        set_status(node_registry, key, 'accepted');
        fprintf('  -> accepted (passed all %d starts)\n', size(all_x0, 2));
        continue;
    end
 
    % Escalation: prima bt (+ fallback su c1), poi max_cg, poi kmax
    spawned_child = false;
    if any_bt_reached
        idx = find(bt_levels > cfg.bt, 1, 'first');
        if ~isempty(idx)
            newcfg = cfg;
            newcfg.bt = bt_levels(idx);
            newkey = make_key(newcfg);
            if ~tested_keys.isKey(newkey)
                created = register_node(node_registry, newkey, key, ...
                    sprintf('bt escalation (bt %d->%d)', cfg.bt, newcfg.bt), newcfg, 'queued');
                if created
                    queue{end+1} = newcfg; %#ok<SAGROW>
                    spawned_child = true;
                    fprintf('  -> enqueued bt escalation: new bt=%d\n', newcfg.bt);
                end
            end
        end
        if cfg.c1 == c1_baseline
            newcfg2 = cfg;
            newcfg2.c1 = c1_levels(end);
            newkey2 = make_key(newcfg2);
            if ~tested_keys.isKey(newkey2)
                created2 = register_node(node_registry, newkey2, key, ...
                    sprintf('c1 fallback (c1 %.0e->%.0e)', cfg.c1, newcfg2.c1), newcfg2, 'queued');
                if created2
                    queue{end+1} = newcfg2; %#ok<SAGROW>
                    spawned_child = true;
                    fprintf('  -> enqueued c1 fallback: c1=%.0e\n', newcfg2.c1);
                end
            end
        end
        if spawned_child
            set_status(node_registry, key, 'escalated');
        else
            set_status(node_registry, key, 'discarded');
        end
        continue;
    end
 
    if any_cg_reached
        idx_cg = find(max_cg_levels > cfg.max_cg, 1, 'first');
        if ~isempty(idx_cg)
            newcfg = cfg;
            newcfg.max_cg = max_cg_levels(idx_cg);
            newkey = make_key(newcfg);
            if ~tested_keys.isKey(newkey)
                created = register_node(node_registry, newkey, key, ...
                    sprintf('max\\_cg escalation (max\\_cg %d->%d)', cfg.max_cg, newcfg.max_cg), newcfg, 'queued');
                if created
                    queue{end+1} = newcfg; %#ok<SAGROW>
                    spawned_child = true;
                    fprintf('  -> enqueued max_cg escalation: new max_cg=%d\n', newcfg.max_cg);
                end
            end
        else
            fprintf('  -> max_cg saturato (nessun livello superiore), scartata\n');
        end
        if spawned_child
            set_status(node_registry, key, 'escalated');
        else
            set_status(node_registry, key, 'discarded');
        end
        continue;
    end
 
    if any_k_reached
        idxk = find(kmax_levels > cfg.kmax, 1, 'first');
        if ~isempty(idxk)
            newcfg = cfg;
            newcfg.kmax = kmax_levels(idxk);
            newkey = make_key(newcfg);
            if ~tested_keys.isKey(newkey)
                created = register_node(node_registry, newkey, key, ...
                    sprintf('kmax escalation (kmax %d->%d)', cfg.kmax, newcfg.kmax), newcfg, 'queued');
                if created
                    queue{end+1} = newcfg; %#ok<SAGROW>
                    spawned_child = true;
                    fprintf('  -> enqueued kmax escalation: new kmax=%d\n', newcfg.kmax);
                end
            end
        else
            fprintf('  -> kmax saturato (nessun livello superiore), scartata\n');
        end
        if spawned_child
            set_status(node_registry, key, 'escalated');
        else
            set_status(node_registry, key, 'discarded');
        end
        continue;
    end
 
    set_status(node_registry, key, 'discarded');
    fprintf('  -> scartata: non converge e nessuna regola di escalation applicabile\n');
end
 
% Normalizzazione difensiva: eventuali nodi rimasti 'queued' (non
% dovrebbe succedere nel flusso normale) vengono marcati come scartati
% cosi' il plot non li lascia in un colore ambiguo.
all_reg_keys = keys(node_registry);
for i = 1:numel(all_reg_keys)
    nd = node_registry(all_reg_keys{i});
    if strcmp(nd.status, 'queued')
        set_status(node_registry, all_reg_keys{i}, 'discarded');
    end
end
 
fprintf('\nRefinement 1 completato. Configurazioni buone trovate: %d\n', numel(good_configs));
 
if ~isempty(good_configs)
    keys_good = arrayfun(make_key, good_configs, 'UniformOutput', false);
    [~, ia] = unique(keys_good, 'stable');
    good_configs = good_configs(ia);
    fprintf('Dopo deduplicazione: %d\n', numel(good_configs));
end
 
fprintf('\n--- Configurazioni testate (riepilogo) ---\n');
tested_keys_list = keys(tested_keys);
for i = 1:numel(tested_keys_list)
    fprintf('%s\n', tested_keys_list{i});
end
 
% Albero delle configurazioni esplorate durante il Refinement 1.
plot_config_tree(node_registry, ...
    sprintf('Truncated Newton (%s) - albero configurazioni Refinement 1', deriv_mode));
 
% ---------------- REFINEMENT 2: valutazione robusta e loss ----------------
fprintf('\n=== REFINEMENT 2: robust evaluation (n = %s; %d starts each) ===\n', ...
    mat2str(n_ref2), n_starts_ref2);
 
refined = [];
for i = 1:numel(good_configs)
    cfg = good_configs(i);
    L_cfg = -Inf;
    success_cfg = true;
 
    fprintf('\nEvaluating config #%d: rho=%g c1=%.0e kmax=%d bt=%d max_cg=%d\n', ...
        i, cfg.rho, cfg.c1, cfg.kmax, cfg.bt, cfg.max_cg);
 
    for n = n_ref2
        x0_base = xbar_gen(n);
        x0_rand = (x0_base - 1) + 2*rand(n, n_starts_ref2 - 1);
        all_x0  = [x0_base, x0_rand];
 
        for s = 1:size(all_x0, 2)
            t = zeros(1, 3);
            for r = 1:3
                tic;
                [~, ~, gradnorm, k, ~, btseq, ~, inner_iters] = truncated_newton_method( ...
                    all_x0(:,s), f, gradf, hessf, ...
                    cfg.kmax, tolgrad, cfg.c1, cfg.rho, cfg.bt, cfg.max_cg);
                t(r) = toc;
            end
            t_run = median(t);
 
            bt_saturated = ~isempty(btseq) && any(btseq >= cfg.bt);
            cg_saturated = ~isempty(inner_iters) && any(inner_iters >= cfg.max_cg);
            
            if gradnorm >= tolgrad && ( k >= cfg.kmax || bt_saturated || cg_saturated)
                fprintf('  run fallita a n=%d start=%d: gradnorm=%.2e k=%d bt_sat=%d cg_sat=%d\n', ...
                    n, s, gradnorm, k, bt_saturated, cg_saturated);
                success_cfg = false;
                break;
            end
 
            phi   = max(0, log10(gradnorm / tolgrad));
            L_run = w_t * t_run + w_g * phi;
            L_cfg = max(L_cfg, L_run);
        end
        if ~success_cfg
            break;
        end
    end
 
    if success_cfg
        r = struct();
        r.max_cg = cfg.max_cg;
        r.rho    = cfg.rho;
        r.c1     = cfg.c1;
        r.kmax   = cfg.kmax;
        r.bt     = cfg.bt;
        r.loss   = L_cfg;
        refined = [refined; r]; %#ok<AGROW>
        fprintf('  mantenuta in REF2: loss=%.4e\n', L_cfg);
    else
        fprintf('  scartata in REF2 (non robusta)\n');
    end
end
 
if isempty(refined)
    error('Nessuna configurazione ha superato il refinement 2.');
end
 
% ---------------- Selezione finale e grafici ----------------
losses = [refined.loss];
pct = 100;
thr = prctile(losses, pct);
best = refined(losses <= thr);
 
fprintf('\n=== SELEZIONE FINALE: top %d%% (loss <= %.4g) ===\n', pct, thr);
for i = 1:numel(best)
    fprintf('%d) rho=%.6g c1=%.0e kmax=%d bt=%d max_cg=%d loss=%.4e\n', ...
        i, best(i).rho, best(i).c1, best(i).kmax, best(i).bt, best(i).max_cg, best(i).loss);
end
 
rho_vals  = [refined.rho];
c1_vals   = [refined.c1];
bt_vals   = [refined.bt];
cg_vals   = [refined.max_cg];
loss_vals = [refined.loss];
 
figure;
scatter(rho_vals, c1_vals, 80, log10(loss_vals), 'filled');
colorbar;
xlabel('\rho'); ylabel('c1');
title(sprintf('Truncated Newton (%s) - color = log_{10}(loss)', deriv_mode));
grid on;
hold on;
for i = 1:numel(best)
    scatter(best(i).rho, best(i).c1, 150, 'k', 'LineWidth', 1.5);
end
hold off;
 
figure;
scatter(bt_vals, cg_vals, 80, log10(loss_vals), 'filled');
colorbar;
xlabel('bt'); ylabel('max\_cg');
title('bt vs max\_cg (log_{10} loss)');
grid on;
 
figure;
scatter(rho_vals, cg_vals, 80, log10(loss_vals), 'filled');
colorbar;
xlabel('\rho'); ylabel('max\_cg');
title('\rho vs max\_cg (log_{10} loss)');
grid on;
 
% ---------------- Albero con le configurazioni finali evidenziate ----------------
best_keys = arrayfun(@(r) sprintf('r%.6g_c%.6g_k%d_bt%d_cg%d', ...
    r.rho, r.c1, r.kmax, r.bt, r.max_cg), best, 'UniformOutput', false);
plot_config_tree(node_registry, ...
    sprintf('Truncated Newton (%s) - albero con selezione finale evidenziata', deriv_mode), ...
    best_keys);
 
% ---------------- PCA sulle configurazioni sopravvissute al Refinement 2 ----------------
% Parametri su scala log10 (spaziano piu' ordini di grandezza): c1, kmax,
% bt, max_cg. rho resta in scala lineare (range gia' limitato, [0,1]).
kmax_vals_ref2 = [refined.kmax]';
 
param_matrix = [rho_vals(:), log10(c1_vals(:)), log10(kmax_vals_ref2), ...
    log10(bt_vals(:)), log10(cg_vals(:))];
param_names  = {'\rho', 'log_{10}(c1)', 'log_{10}(kmax)', 'log_{10}(bt)', 'log_{10}(max\_cg)'};
 
pca_config_plot(param_matrix, param_names, loss_vals(:), ...
    sprintf('Truncated Newton (%s) - PCA sui parametri (Refinement 2)', deriv_mode));
 
fprintf('\nScript terminato. Configurazioni iniziali=%d, buone dopo ref1=%d, dopo ref2=%d, selezionate=%d\n', ...
    numel(rho_levels), numel(good_configs), numel(refined), numel(best));
