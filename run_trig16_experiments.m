clear; clc; close all;

% Aggiunge questa cartella e TUTTE le sottocartelle al path
project_root = fileparts(mfilename('fullpath'));   % cartella dove sta questo script
addpath(genpath(project_root));



% --- Problem Loading ---
% problem_trig16 deve restituire 5 output, analogamente a problem_broyden31
[f, grad_exact, hess_exact, xbarfun, xstarfun] = problem_trig16();

% --- Fixed Parameters ---
seed    = 346710;
rng(seed);

tolgrad = 1e-6;

n_list  = [2,1e3,1e4,1e5];      % Dimensioni richieste
k_list  = [4, 8, 12];         % Valori di k per le differenze finite
fdtypes = [1];             % 1 = passo costante (h), 2 = passo relativo (hi)

% Parametri tarati per Modified e Truncated Newton
params_modified.kmax = 200;  params_modified.c1 = 1e-4;
params_modified.rho  = 0.5; params_modified.btmax = 30;
params_modified.beta = 1;

params_truncated.kmax = 500;  params_truncated.c1 = 1e-3;
params_truncated.rho  = 0.5; params_truncated.btmax = 40;
params_truncated.max_cg = 1000;

% --- Main Loop ---
results = struct([]);
idx = 0;

for n = n_list
    fprintf('\n========== n = %d ==========\n', n);

    % Generate starting points
    xb = xbarfun(n); % Punto di partenza standard (tutti 1)
    % 5 punti random nell'ipercubo [xb-1, xb+1]
    X0 = [xb, xb + (2*rand(n,5) - 1)];

    % Test cases: Exact, FD Hessian only, Full FD
    for dm = ["case1", "case2"]

    % Determine which k and type values to loop over
    if dm == "exact"
        k_loop = [NaN];
        type_loop = [0];
    else
        k_loop = k_list;
        type_loop = fdtypes;
    end

    for kk = k_loop
        for type = type_loop

            % --- Dynamic function selection ---
            switch dm
                case "exact"
                    gradf = grad_exact;
                    hessf = hess_exact;
                case "case1"
                    % Gradiente esatto, Hessiana approssimata per FD (ibrido)
                    gradf = grad_exact;
                    hessf = @(x) trig_hess_fd_case1(grad_exact, x, kk, type);
                case "case2"
                    % FD completo: gradiente e Hessiana approssimati per FD
                    gradf = @(x) trig_fd_case2_grad_only(x, kk, type);
                    hessf = @(x) trig_fd_case2_hess_only(x, kk, type);
            end

            % Loop over optimization methods
            for method = ["truncated"]
                if method == "modified"
                    prm = params_modified;
                else
                    prm = params_truncated;
                end

                % Loop over the 6 starting points (1 standard + 5 random)
                for s = 1:6
                    x0 = X0(:, s);
                    tic;
                    run_failed = false;
                    try
                        if method == "modified"
                            [~, fk, gn, it, xseq] = modified_newton_method(x0, f, ...
                                gradf, hessf, prm.kmax, tolgrad, prm.c1, prm.rho, ...
                                prm.btmax, prm.beta);
                        else
                            [~, fk, gn, it, xseq,~,~,inner] = truncated_hess_free(x0, f, ...
                                gradf, prm.kmax, tolgrad, prm.c1, prm.rho, ...
                                prm.btmax, prm.max_cg, 10^-kk);
                        end
                    catch ME
                        fk = NaN; gn = Inf; it = 0; xseq = [];
                        run_failed = true;
                        warning('Failed (%s, %s, n=%d, k=%g, type=%d): %s', ...
                            char(method), char(dm), n, kk, type, ME.message);
                    end
                    t = toc;

                    % --- Success flag e diagnosi del motivo di arresto ---
                    % Euristica basata su cio' che le funzioni restituiscono
                    % (it, gn, kmax): non c'e' un output diretto tipo
                    % "backtracking esaurito", quindi si deduce per
                    % esclusione (arrestato prima di kmax, senza aver
                    % raggiunto la tolleranza sul gradiente).
                    succ = ~run_failed && gn < tolgrad;
                    if run_failed
                        stop_reason = "error";
                    elseif succ
                        stop_reason = "converged";
                    elseif it >= prm.kmax
                        stop_reason = "maxit";
                    else
                        stop_reason = "backtracking_or_other";
                    end

                    % Rate calcolato SOLO se la sequenza e' convergente
                    % (sempre col gradiente esatto, per coerenza).
                    if succ
                        gnorm_last3 = zeros(1,3);
                        K = size(xseq,2);
                        for i = 1:3
                            gnorm_last3(i) = norm(grad_exact(xseq(:, K-3+i)));
                        end
                        [rate_seq, ~] = estimate_rate_polyfit(gnorm_last3, 3);
                        rate = rate_seq; % un solo valore, la finestra e' l'intera sequenza di 4 punti
                    else
                        rate = NaN;
                    end

                    % Save results
                    idx = idx + 1;
                    results(idx).n = n;
                    results(idx).deriv_mode = dm;
                    results(idx).k = kk;
                    results(idx).type = type;
                    results(idx).method = method;
                    results(idx).start = s;
                    results(idx).iter = it;
                    results(idx).gradnorm = gn;
                    results(idx).success = succ;
                    results(idx).stop_reason = stop_reason;
                    results(idx).rate = rate;
                    results(idx).time = t;
                    results(idx).xseq = xseq;

                    % --- Safe formatting for printing ---
                    if isnan(kk)
                        kstr = "n/a";
                        typestr = "n/a";
                    else
                        kstr = string(kk);
                        if type == 1, typestr = "h"; else, typestr = "hi"; end
                    end

                    fprintf('%-9s | %-6s | n=%-6d | k=%-4s | %-3s | start=%d | iter=%3d | gn=%.2e | succ=%d | reason=%-22s | rate=%.2f | t=%.2fs\n', ...
                        method, dm, n, kstr, typestr, s, it, gn, succ, stop_reason, rate, t);
                end
            end
        end
    end
end
end

% --- Save Results ---
save('trig16_fd_results_MODIFIED.mat', 'results');
disp('Done. Results saved in trig16_fd_results.mat');

% --- Riepilogo run non convergenti, per motivo ---
non_conv = ~[results.success];
if any(non_conv)
    reasons = string({results(non_conv).stop_reason});
    fprintf('\n--- Run NON convergenti: %d / %d ---\n', nnz(non_conv), numel(results));
    for r = unique(reasons)
        fprintf('  %-22s : %d\n', r, sum(reasons == r));
    end
end


%% --- Generazione automatica dei grafici per TUTTE le combinazioni ---
% Stessa logica di run_experiments_31.m: per ogni n, deriv_mode
% (e k/type quando applicabile) e method, genera:
%   - plot_error_ratio             (sempre, tutte le run)
%   - plot_contour_paths           (solo per n = 2, tutte le run)
%   - plot_convergence_rate        (solo run convergenti)
%   - plot_convergence_rate_trig16 (solo run convergenti)
%   - plot_error_to_xstar          (sempre, tutte le run)
%
% I grafici finiscono in graphs_trig16/n<N>/, con nomi di file
% univoci tipo:
%   TruncatedNewton_case2_k8_hi_n1000_...
%   ModifiedNewton_exact_n2_...
method_map = containers.Map({'modified','truncated'}, ...
                             {'ModifiedNewton','TruncatedNewton'});
 
base_outdir = 'graphs_trig16_flippedGiusti';
 
set(0, 'DefaultFigureVisible', 'off');
 
for n_target = n_list
 
    %xstar_n = xstarfun(n_target);
    outdir_n = fullfile(base_outdir, sprintf('n%d', n_target));
 
    outdir_contour       = fullfile(outdir_n, 'contour');
    outdir_rate1          = fullfile(outdir_n, 'convergence_rate1');
    outdir_rate3          = fullfile(outdir_n, 'convergence_rate3');
    outdir_rate5          = fullfile(outdir_n, 'convergence_rate5');
    outdir_rate_iterdiff1 = fullfile(outdir_n, 'convergence_rate_iterdiff1');
    outdir_rate_iterdiff3 = fullfile(outdir_n, 'convergence_rate_iterdiff3');
    outdir_rate_iterdiff5 = fullfile(outdir_n, 'convergence_rate_iterdiff5');
    outdir_errratio      = fullfile(outdir_n, 'error_ratio');
    outdir_errxstar      = fullfile(outdir_n, 'error_to_xstar');

    % Sottocartelle per i grafici AGGREGATI (tutti i k/type in un unico
    % plot, colorati per gruppo), window=3 e window=5. NIENTE variante
    % xstar qui: su Trig16 (periodica, minimi multipli equivalenti) la
    % distanza da un singolo x* non e' un indicatore affidabile, motivo
    % per cui anche plot_error_to_xstar/plot_error_ratio restano
    % commentate poco sotto - restiamo coerenti anche nei grafici
    % aggregati e usiamo solo grad e iterdiff.
    %outdir_rate_grouped1          = fullfile(outdir_n, 'convergence_rate_grouped1');
    %outdir_rate_grouped3          = fullfile(outdir_n, 'convergence_rate_grouped3');
    %outdir_rate_grouped5          = fullfile(outdir_n, 'convergence_rate_grouped5');

    %outdir_rate_grouped_iterdiff1 = fullfile(outdir_n, 'convergence_rate_grouped_iterdiff1');
    %outdir_rate_grouped_iterdiff3 = fullfile(outdir_n, 'convergence_rate_grouped_iterdiff3');
    %outdir_rate_grouped_iterdiff5 = fullfile(outdir_n, 'convergence_rate_grouped_iterdiff5');
 
    for dm = ["exact", "case1", "case2"]
 
        if dm == "exact"
            k_loop_plot    = NaN;
            type_loop_plot = 0;
        else
            k_loop_plot    = k_list;
            type_loop_plot = fdtypes;
        end
 
        for kk = k_loop_plot
            for type = type_loop_plot
 
                for method = ["modified", "truncated"]
 
                    method_label = method_map(char(method));
 
                    mask = [results.n] == n_target & ...
                            [results.deriv_mode] == dm & ...
                            [results.method] == method;
 
                    if dm ~= "exact"
                        if isnan(kk)
                            mask = mask & isnan([results.k]);
                        else
                            mask = mask & [results.k] == kk;
                        end
                        mask = mask & [results.type] == type;
                    end
 
                    if ~any(mask)
                        continue;
                    end
 
                    xseq_list = {results(mask).xseq};
                    if all(cellfun(@isempty, xseq_list))
                        continue;
                    end

                    % xseq_list filtrata sui SOLI run convergenti, usata
                    % unicamente per i due plot di rate.
                    mask_conv = mask & [results.success];
                    xseq_list_conv = {results(mask_conv).xseq};
 
                    if dm == "exact"
                        tag = sprintf('%s_exact_n%d', method_label, n_target);
                    else
                        if type == 1, typestr = 'h'; else, typestr = 'hi'; end
                        tag = sprintf('%s_%s_k%d_%s_n%d', method_label, dm, kk, typestr, n_target);
                    end
 
                    fprintf('--- Grafici per: %s ---\n', tag);
 
                    %plot_error_ratio(xseq_list, xstar_n, tag, 'Trig16', outdir_errratio, 4);
 
                    if n_target == 2
                        plot_contour_paths(f, xseq_list, tag, 'Trig16', outdir_contour);
                    end
 
                    if ~isempty(xseq_list_conv) && ~all(cellfun(@isempty, xseq_list_conv))
                        plot_convergence_rate(xseq_list_conv, grad_exact, tag, 'Trig16', outdir_rate1, 1);
                        plot_convergence_rate(xseq_list_conv, grad_exact, tag, 'Trig16', outdir_rate3, 3);
                        plot_convergence_rate(xseq_list_conv, grad_exact, tag, 'Trig16', outdir_rate5, 5);
                        %plot_convergence_rate_trig16(xseq_list_conv, tag, 'Trig16', outdir_rate_iterdiff1, 1);
                        %plot_convergence_rate_trig16(xseq_list_conv, tag, 'Trig16', outdir_rate_iterdiff3, 3);
                        %plot_convergence_rate_trig16(xseq_list_conv, tag, 'Trig16', outdir_rate_iterdiff5, 5);
                    else
                        fprintf('  (nessuna run convergente per %s: rate plots saltati)\n', tag);
                    end
 
                    %plot_error_to_xstar(xseq_list, xstar_n, tag, 'Trig16', outdir_errxstar);
 
                    close all;
 
                end
            end
        end
    end

end
disp('Tutti i grafici sono stati generati e salvati in graphs_trig16/n<N>/<tipo_grafico>/');