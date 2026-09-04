clear; clc; close all;

% ============================================================
% Confronto generalizzato: correzione Hessiana Hk+tau*I (originale)
% vs flipping degli autovalori, per il problema Trig16.
%
% Estende la logica di run_trig16_experiments.m: stesso schema di loop
% (n, deriv_mode = exact/case1/case2, k, type, 6 starting point), ma per
% OGNI combinazione lancia ENTRAMBI i solver sugli STESSI punti di
% partenza, cosi' il confronto e' diretto (stessa condizione iniziale,
% stessa Hessiana approssimata quando applicabile).
%
% Salva un unico results struct array con i risultati di entrambe le
% varianti fianco a fianco, e produce alla fine tabelle aggregate di
% confronto (tassi di successo, iterazioni medie, tempo, numero di
% correzioni/flip usati).
%
% NOTA: il contour plot (intrinsecamente 2D) viene generato SOLO per
% n_target == 2, come in run_trig16_experiments.m.
% ============================================================

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbar_gen] = problem_trig16();

seed    = 346710;
tolgrad = 1e-6;

n_list    = [2, 1e3, 1e4,1e5];   % aggiungere 1e5 se il flip regge quella scala (vedi nota sotto)
k_list    = [4, 8, 12];
fdtypes   = [1, 2];

% Parametri comuni ad entrambi i solver (stessi per un confronto equo)
kmax  = 200;
c1    = 1e-4;
rho   = 0.5;
btmax = 30;
beta  = 0.01;   % solo per la variante tau*I

% ATTENZIONE SU n GRANDI: se modified_newton_method_flip corregge la
% Hessiana con un flip degli AUTOVALORI (es. via eig/eigendecomposizione
% densa), quell'operazione non scala a n=1e5 (costo O(n^3) e memoria
% O(n^2) per matrice piena). Verificare come e' implementata la
% correzione prima di aggiungere n grandi a n_list: se usa una
% decomposizione sparsa/strutturata (sfruttando che l'Hessiana esatta di
% Trig16 e' diagonale) allora n grandi sono ok, altrimenti limitare
% n_list a valori piccoli/medi.

results = struct([]);
idx = 0;

for n = n_list
    fprintf('\n========== n = %d ==========\n', n);

    rng(seed + n, 'twister');
    xb = xbar_gen(n);
    X0 = [xb, xb + (2*rand(n,5) - 1)];

    for dm = ["exact", "case1", "case2"]

        if dm == "exact"
            k_loop = NaN; type_loop = 0;
        else
            k_loop = k_list; type_loop = fdtypes;
        end

        for kk = k_loop
            for type = type_loop

                switch dm
                    case "exact"
                        gradf = grad_exact;
                        hessf = hess_exact;
                    case "case1"
                        gradf = grad_exact;
                        hessf = @(x) trig_hess_fd_case1(grad_exact, x, kk, type);
                    case "case2"
                        gradf = @(x) trig_fd_case2_grad_only(x, kk, type);
                        hessf = @(x) trig_fd_case2_hess_only(x, kk, type);
                end

                for s = 1:6
                    x0 = X0(:, s);

                    % --- Variante originale: Hk + tau*I ---
                    tic;
                    try
                        [~, fk, gn, it, xseq,~,~,inner] = truncated_hess_free(x0, f, ...
                            gradf, prm.kmax, tolgrad, prm.c1, prm.rho, ...
                            prm.btmax, prm.max_cg);
                    catch ME
                        fk_tau = NaN; gn_tau = Inf; k_tau = 0; xseq_tau = [];
                        warning('tau*I fallito (n=%d,%s,k=%g,type=%d,s=%d): %s', ...
                            n, dm, kk, type, s, ME.message);
                    end
                    t_tau = toc;

                    % --- Variante flip autovalori ---
                    tic;
                    try
                        [~, fk_flip, gn_flip, k_flip, xseq_flip, ~, ~, ~, ~, n_flips] = ...
                            modified_newton_method_flip( ...
                            x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax);
                    catch ME
                        fk_flip = NaN; gn_flip = Inf; k_flip = 0; xseq_flip = []; n_flips = NaN;
                        warning('flip fallito (n=%d,%s,k=%g,type=%d,s=%d): %s', ...
                            n, dm, kk, type, s, ME.message);
                    end
                    t_flip = toc;

                    succ_tau  = gn_tau  < tolgrad;
                    succ_flip = gn_flip < tolgrad;

                    rate_tau  = estimate_rate(xseq_tau,  grad_exact);
                    rate_flip = estimate_rate(xseq_flip, grad_exact);

                    idx = idx + 1;
                    results(idx).n      = n;
                    results(idx).dm     = char(dm);   % char, non string: serve per strcmp piu' avanti
                    results(idx).k      = kk;
                    results(idx).type   = type;
                    results(idx).start  = s;

                    results(idx).fk_tau     = fk_tau;
                    results(idx).gn_tau     = gn_tau;
                    results(idx).iter_tau   = k_tau;
                    results(idx).succ_tau   = succ_tau;
                    results(idx).time_tau   = t_tau;
                    results(idx).rate_tau   = rate_tau;
                    results(idx).xseq_tau   = xseq_tau;

                    results(idx).fk_flip    = fk_flip;
                    results(idx).gn_flip    = gn_flip;
                    results(idx).iter_flip  = k_flip;
                    results(idx).succ_flip  = succ_flip;
                    results(idx).time_flip  = t_flip;
                    results(idx).rate_flip  = rate_flip;
                    results(idx).n_flips    = n_flips;
                    results(idx).xseq_flip  = xseq_flip;

                    % --- metriche di confronto dirette ---
                    results(idx).iter_diff     = k_flip - k_tau;
                    results(idx).time_ratio    = t_flip / t_tau;
                    results(idx).gradnorm_ratio = gn_flip / gn_tau;

                    if isnan(kk)
                        kstr = "n/a"; typestr = "n/a";
                    else
                        kstr = string(kk);
                        if type == 1, typestr = "h"; else, typestr = "hi"; end
                    end

                    fprintf('n=%-6d %-6s k=%-4s %-3s s=%d | tau: it=%3d gn=%.2e succ=%d t=%.2fs | flip: it=%3d gn=%.2e succ=%d flips=%3s t=%.2fs\n', ...
                        n, dm, kstr, typestr, s, ...
                        k_tau,  gn_tau,  succ_tau,  t_tau, ...
                        k_flip, gn_flip, succ_flip, string(n_flips), t_flip);
                end
            end
        end
    end
end

save('trig16_flip_vs_tau_results.mat', 'results');
disp('Salvato in trig16_flip_vs_tau_results.mat');


%% ============================================================
% TABELLE AGGREGATE DI CONFRONTO
% Raggruppate per (n, deriv_mode): media su tutte le combinazioni di
% k/type/starting point.
% ============================================================
fprintf('\n\n===============================================\n');
fprintf(' RIEPILOGO AGGREGATO (media su k, type, starting point)\n');
fprintf('===============================================\n');
fprintf('%-8s %-7s | %-22s | %-22s | %-10s | %-18s | %-18s\n', 'n', 'dm', ...
    'tau: succ%% / iter / time', 'flip: succ%% / iter / time', 'flips medi', ...
    'rate tau (n_val)', 'rate flip (n_val)');
 
for n = n_list
    for dm = ["exact", "case1", "case2"]
        mask = [results.n] == n & strcmp({results.dm}, char(dm));
        if ~any(mask), continue; end
 
        R = results(mask);
 
        succ_tau_pct  = 100*mean([R.succ_tau]);
        succ_flip_pct = 100*mean([R.succ_flip]);
        iter_tau_mean  = mean([R.iter_tau]);
        iter_flip_mean = mean([R.iter_flip]);
        time_tau_mean  = mean([R.time_tau]);
        time_flip_mean = mean([R.time_flip]);
        flips_mean = mean([R.n_flips], 'omitnan');
 
        % Rate: estimate_rate spesso restituisce NaN (scarta i casi
        % rumorosi/non convergenti), quindi si riporta la media SOLO
        % sui run validi, insieme al conteggio (n_validi / n_totali):
        % una media su pochissimi dati non-NaN sarebbe fuorviante senza
        % sapere quanti punti la sostengono.
        rate_tau_vals  = [R.rate_tau];
        rate_flip_vals = [R.rate_flip];
        n_valid_tau  = sum(~isnan(rate_tau_vals));
        n_valid_flip = sum(~isnan(rate_flip_vals));
        rate_tau_mean  = mean(rate_tau_vals,  'omitnan');
        rate_flip_mean = mean(rate_flip_vals, 'omitnan');
 
        fprintf('%-8d %-7s | %5.1f%% / %6.1f / %5.2fs | %5.1f%% / %6.1f / %5.2fs | %8.2f | %6.2f (%3d/%-3d)   | %6.2f (%3d/%-3d)\n', ...
            n, dm, ...
            succ_tau_pct, iter_tau_mean, time_tau_mean, ...
            succ_flip_pct, iter_flip_mean, time_flip_mean, ...
            flips_mean, ...
            rate_tau_mean,  n_valid_tau,  numel(R), ...
            rate_flip_mean, n_valid_flip, numel(R));
    end
end
 
% --- Conteggio "vittorie": chi converge quando l'altro non converge ---
fprintf('\n--- Casi in cui SOLO una variante converge ---\n');
only_tau  = sum([results.succ_tau] & ~[results.succ_flip]);
only_flip = sum(~[results.succ_tau] & [results.succ_flip]);
both      = sum([results.succ_tau] & [results.succ_flip]);
neither   = sum(~[results.succ_tau] & ~[results.succ_flip]);
fprintf('Solo tau*I converge : %d / %d run\n', only_tau, numel(results));
fprintf('Solo flip converge  : %d / %d run\n', only_flip, numel(results));
fprintf('Entrambe convergono : %d / %d run\n', both, numel(results));
fprintf('Nessuna converge    : %d / %d run\n', neither, numel(results));
 
 
% ============================================================
% Boxplot di confronto dei rate stimati (tau vs flip), per deriv_mode,
% pooling su tutti gli n/k/type/starting point. Solo i valori non-NaN
% vengono inclusi (vedi nota sopra sul perche' estimate_rate produce
% spesso NaN).
% ============================================================
figure('Color','w', 'Position', [50 50 1200 450]);
dm_list = ["exact", "case1", "case2"];
for i_dm = 1:numel(dm_list)
    dm = dm_list(i_dm);
    mask_dm = strcmp({results.dm}, char(dm));
    R = results(mask_dm);
 
    rt = [R.rate_tau];  rt = rt(~isnan(rt));
    rf = [R.rate_flip]; rf = rf(~isnan(rf));
 
    subplot(1, numel(dm_list), i_dm);
    if isempty(rt) && isempty(rf)
        text(0.5, 0.5, 'nessun rate valido', 'HorizontalAlignment','center');
        axis off;
    else
        groups = [ones(numel(rt),1); 2*ones(numel(rf),1)];
        vals   = [rt(:); rf(:)];
        boxplot(vals, groups, 'Labels', {sprintf('tau (n=%d)',numel(rt)), sprintf('flip (n=%d)',numel(rf))});
        ylim([0 3]);
        grid on;
    end
    title(char(dm), 'Interpreter','none');
end
sgtitle('Confronto rate stimato (estimate rate): tau*I vs flip, per deriv_mode');
 
 
% ============================================================
% Contour plot di confronto (solo n = 2, come in run_trig16_experiments)
% ============================================================
n_target = 2;
outdir ='graphs_trig16_FLIPPED'; 
if any(n_list == n_target)
    mask2 = [results.n] == n_target & strcmp({results.dm}, 'exact');
    R2 = results(mask2);
 
    if isempty(R2)
        warning('Nessun risultato trovato per n=2, dm=exact: contour plot saltato.');
    else
        xseq_list_tau  = {R2.xseq_tau};
        xseq_list_flip = {R2.xseq_flip};
 
        plot_contour_paths(f, xseq_list_tau,  'ModifiedNewton_tauI', 'Trig16', outdir);
        plot_contour_paths(f, xseq_list_flip, 'ModifiedNewton_flip', 'Trig16', outdir);
        disp('Contour plot (n=2, exact) salvati in graphs_trig16_flip_compare/');
    end
end