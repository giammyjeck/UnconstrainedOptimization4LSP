clear; clc; close all;

% Aggiunge questa cartella e TUTTE le sottocartelle al path
project_root = fileparts(mfilename('fullpath'));   % cartella dove sta questo script
addpath(genpath(project_root));


% --- Problem Loading ---
[f, grad_exact, hess_exact, xbarfun, rfun,xstarfun] = problem_broyden31();

% --- Fixed Parameters ---
seed    = 346710;
rng(seed);
tolgrad = 1e-6;

n_list  = [2,1e3,1e4,1e5]; % Required dimensions[cite: 2]
k_list  = [4,8,12];         % Required k values for FD[cite: 2]
fdtypes = [1];             % 1 = constant step (h), 2 = relative step (hi)

% Tuned Parameters for Modified and Truncated Newton
% (These might need adjustment for n = 1e5 if the algorithm struggles)
params_modified.kmax = 50;  params_modified.c1 = 1e-3;
params_modified.rho  = 0.5; params_modified.btmax = 30;
params_modified.beta = 1e-2;

params_truncated.kmax = 50;  params_truncated.c1 = 1e-4;
params_truncated.rho  = 0.3; params_truncated.btmax = 10;
params_truncated.max_cg = 100;

% --- Main Loop ---
results = struct([]);
idx = 0;

for n = n_list
    fprintf('\n========== n = %d ==========\n', n);

    % Generate starting points
    xb = xbarfun(n); % Standard starting point (all -1)[cite: 1, 2]
    % Generate 5 random points in the hypercube [-2, 0]
    % Since xb = -1, xb + (2*rand - 1) covers [-2, 0][cite: 1, 2]
    X0 = [xb, xb + (2*rand(n,5) - 1)];   
    %xstar_n = xstarfun(n);
    % Test cases: Exact, FD Hessian only, Full FD
    for dm = ["case1", "case2"]
    
    % Determine which k and type values to loop over
    if dm == "exact"
        k_loop = [NaN];      % Single dummy value (no FD loop needed)
        type_loop = [0];     % Single dummy value (no type loop needed)
    else
        k_loop = k_list;     % Loop over k = [4, 8, 12]
        type_loop = fdtypes; % Loop over type = [1, 2]
    end
    
    % Loop over k values
    for kk = k_loop
        
        % Loop over type values (constant step vs relative step)
        for type = type_loop
            
            % --- Dynamic function selection ---
            switch dm
                case "exact"
                    gradf = grad_exact;
                    hessf = hess_exact;
                case "case1"
                    % Exact gradient, FD Hessian (hybrid)
                    gradf = grad_exact;
                    hessf = @(x) hess_fd_broyden31(grad_exact, x, kk, type);
                case "case2"
                    % Full FD: FD gradient and FD Hessian
                    gradf = @(x) grad_fd_broyden31(x, kk, type, rfun);
                    hessf = @(x) hess_grad_fd_broyden31(x, kk, type, rfun);
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
                                prm.btmax, prm.max_cg,10^-kk);
                        end
                    catch ME
                        fk = NaN; gn = Inf; it = 0; xseq = [];
                        run_failed = true;
                        warning('Failed (%s, %s, n=%d, k=%g, type=%d): %s', ...
                            char(method), char(dm), n, kk, type, ME.message);
                    end
                    t = toc;

                    % --- Success flag e diagnosi del motivo di arresto ---
                    % NB: e' un'euristica basata su quanto le funzioni
                    % restituiscono (it, gn, kmax): modified/truncated
                    % non espongono direttamente "mi sono fermato per
                    % backtracking esaurito", quindi lo si deduce per
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

                    % Rate: calcolato SOLO se la sequenza e' convergente.
                    % Il rate su una sequenza che non converge non ha un
                    % significato affidabile (potrebbe essere stata
                    % interrotta a meta' transitorio, con backtracking
                    % pesante o correzione dell'Hessiana ancora attiva),
                    % e finora inquinava sia la tabella sia i grafici.
                    if succ
                        gnorm_last5 = zeros(1,5);
                        K = size(xseq,2);
                        for i = 1:5
                            gnorm_last5(i) = norm(grad_exact(xseq(:, K-5+i)));
                        end
                        [rate_seq, ~] = estimate_rate_polyfit(gnorm_last5, 5);
                        rate = rate_seq; % un solo valore, la finestra e' l'intera sequenza di 5 punti
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
                    %results(idx).inner_iters = inner;
                    
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
save('broyden31_fd_results.mat', 'results');
disp('Done. Results saved in broyden31_fd_results.mat');

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
% Per ogni n, deriv_mode (e k/type quando applicabile) e method, genera:
%   - plot_error_ratio        (sempre, tutte le run)
%   - plot_contour_paths      (solo per n = 2, tutte le run)
%   - plot_convergence_rate   (solo run convergenti)
%   - plot_convergence_rate_trig16 (rate su iterate successive, senza
%     bisogno di xstar; solo run convergenti - vedi nota sotto)
%   - plot_error_to_xstar     (sempre, tutte le run)
%
% I grafici finiscono in graphs_broyden31/n<N>/ , con nomi di file
% univoci ed espliciti tipo:
%   TruncatedNewton_case2_k8_hi_n1000_...
%   ModifiedNewton_exact_n2_...
%
% method_map traduce l'etichetta usata internamente in results (stringhe
% "modified"/"truncated") nel nome leggibile da usare nei titoli/file.
method_map = containers.Map({'modified','truncated'}, ...
                             {'ModifiedNewton','TruncatedNewton'});
 
base_outdir = 'graphs_broyden31_TRUNCSENZADK_newww';
 
% Disabilita la visualizzazione a schermo delle figure: vengono create
% e salvate su disco senza aprire finestre (utile con decine di
% combinazioni). Viene ripristinato il comportamento di default alla
% fine del blocco, anche in caso di errore (onCleanup).
old_visible_state = get(0, 'DefaultFigureVisible');
set(0, 'DefaultFigureVisible', 'off');
restore_visibility = onCleanup(@() set(0, 'DefaultFigureVisible', old_visible_state));
 
for n_target = n_list
 
    xstar_n = xstarfun(n_target);
    outdir_n = fullfile(base_outdir, sprintf('n%d', n_target));
 
    % Sottocartelle per tipo di grafico
    outdir_contour       = fullfile(outdir_n, 'contour');
    outdir_rate3          = fullfile(outdir_n, 'convergence_rate3');
    outdir_rate5         = fullfile(outdir_n, 'convergence_rate5');
    outdir_rate_iterdiff3 = fullfile(outdir_n, 'convergence_rate_iterdiff3');
    outdir_rate_iterdiff5 = fullfile(outdir_n, 'convergence_rate_iterdiff5');
    outdir_rate_xstar3    = fullfile(outdir_n, 'convergence_rate_xstar3');
    outdir_rate_xstar5    = fullfile(outdir_n, 'convergence_rate_xstar5');
    outdir_errratio       = fullfile(outdir_n, 'error_ratio');
    outdir_errxstar       = fullfile(outdir_n, 'error_to_xstar');

    % Sottocartelle per i grafici AGGREGATI (tutti i k/type in un unico
    % plot, colorati per gruppo), separate per window=3 e window=5, e
    % per le tre varianti di errore (grad, iterdiff, xstar) - stesso
    % schema di naming delle altre, con suffisso 3/5.
    outdir_rate_grouped3          = fullfile(outdir_n, 'convergence_rate_grouped3');
    outdir_rate_grouped5          = fullfile(outdir_n, 'convergence_rate_grouped5');
    outdir_rate_grouped_iterdiff3 = fullfile(outdir_n, 'convergence_rate_grouped_iterdiff3');
    outdir_rate_grouped_iterdiff5 = fullfile(outdir_n, 'convergence_rate_grouped_iterdiff5');
    outdir_rate_grouped_xstar3    = fullfile(outdir_n, 'convergence_rate_grouped_xstar3');
    outdir_rate_grouped_xstar5    = fullfile(outdir_n, 'convergence_rate_grouped_xstar5');

 
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
 
                    % --- maschera sui risultati per questa combinazione ---
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
                        continue; % combinazione non presente nei risultati
                    end
 
                    xseq_list = {results(mask).xseq};
                    if all(cellfun(@isempty, xseq_list))
                        continue; % tutti i run sono falliti (xseq vuoto)
                    end

                    % xseq_list filtrata sui SOLI run convergenti, usata
                    % unicamente per i due plot di rate.
                    mask_conv = mask & [results.success];
                    xseq_list_conv = {results(mask_conv).xseq};
 
                    % --- tag univoco ed esplicativo per titoli/nomi file ---
                    if dm == "exact"
                        tag = sprintf('%s_exact_n%d', method_label, n_target);
                    else
                        if type == 1, typestr = 'h'; else, typestr = 'hi'; end
                        tag = sprintf('%s_%s_k%d_%s_n%d', method_label, dm, kk, typestr, n_target);
                    end
 
                    fprintf('--- Grafici per: %s ---\n', tag);
 
                    % Nota: passiamo "tag" (gia' univoco: metodo + deriv_mode
                    % + k/type + n) come method_name, e 'Broyden31' come
                    % problem_name fisso. Cosi' il nome file composto da
                    % ciascuna funzione (<prefisso>_<method>_<problem>)
                    % resta univoco senza duplicare informazioni. Ogni tipo
                    % di grafico va nella sua sottocartella.
                    plot_error_ratio(xseq_list, xstar_n, tag, 'Broyden31', outdir_errratio, 4);
 
                    if n_target == 2
                        plot_contour_paths(f, xseq_list, tag, 'Broyden31', outdir_contour);
                    end
 
                    if ~isempty(xseq_list_conv) && ~all(cellfun(@isempty, xseq_list_conv))
                        plot_convergence_rate(xseq_list_conv, grad_exact, tag, 'Broyden31', outdir_rate3,1);
                        %plot_convergence_rate(xseq_list_conv, grad_exact, tag, 'Broyden31', outdir_rate5,3);
                        % Aggiunto anche qui: rate stimato sul rapporto fra
                        % iterazioni successive (||x_k - x_{k-1}||), senza
                        % bisogno di xstar. Nome della funzione lasciato
                        % invariato (nato per trig16) ma non c'e' nulla di
                        % specifico a quel problema al suo interno.
                        plot_convergence_rate_trig16(xseq_list_conv, tag, 'Broyden31', outdir_rate_iterdiff5, 1);
                        %plot_convergence_rate_xstar(xseq_list_conv, xstar_n, tag, 'Broyden31', outdir_rate_xstar3, 3);
                        plot_convergence_rate_xstar(xseq_list_conv, xstar_n, tag, 'Broyden31', outdir_rate_xstar5, 1);
                    else
                        fprintf('  (nessuna run convergente per %s: rate plots saltati)\n', tag);
                    end
 
                    plot_error_to_xstar(xseq_list, xstar_n, tag, 'Broyden31', outdir_errxstar);
 
                    close all; % evita di accumulare decine di figure in memoria
 
                end
            end
        end
    end

    %% --- Grafici AGGREGATI: tutte le combinazioni k/type in un'unica figura ---
    % Per case1 e case2 (non ha senso per "exact", che non ha k/type),
    % per ogni method, mette in un solo grafico le curve di rate di
    % TUTTI i k={4,8,12} x type={h,hi}, con lo stesso colore per tutte
    % le run che condividono lo stesso (k,type) e colore diverso fra
    % combinazioni diverse. Generati sia con window=3 che window=5,
    % stesso schema delle altre coppie di cartelle *3/*5.
    for dm = ["case1", "case2"]
        for method = ["modified", "truncated"]

            method_label = method_map(char(method));
            base_tag = sprintf('%s_%s_n%d', method_label, dm, n_target);

            % --- costruzione di err_list / group_labels su TUTTI i k/type ---
            % Costruita UNA VOLTA sola (indipendente da window): le due
            % versioni (window 3 e 5) vengono poi ottenute richiamando
            % plot_convergence_rate_grouped due volte sugli stessi dati,
            % senza ricalcolare gnorm/iterdiff/xstar due volte.
            err_list_grad    = {};  group_labels_grad    = {};
            err_list_iterdiff= {};  group_labels_iterdiff= {};
            err_list_xstar   = {};  group_labels_xstar   = {};

            for kk = k_list
                for type = fdtypes

                    mask = [results.n] == n_target & ...
                            [results.deriv_mode] == dm & ...
                            [results.method] == method & ...
                            [results.k] == kk & ...
                            [results.type] == type & ...
                            [results.success];

                    if ~any(mask), continue; end

                    xs = {results(mask).xseq};
                    if type == 1, typestr = 'h'; else, typestr = 'hi'; end
                    lbl = sprintf('k%d_%s', kk, typestr);

                    for i = 1:numel(xs)
                        xseq_i = xs{i};
                        if isempty(xseq_i) || size(xseq_i,2) < 3, continue; end

                        K = size(xseq_i,2);

                        % errore basato sul gradiente (esatto, per coerenza)
                        gnorm = zeros(1,K);
                        for jj = 1:K
                            gnorm(jj) = norm(grad_exact(xseq_i(:,jj)));
                        end
                        err_list_grad{end+1}      = gnorm; %#ok<AGROW>
                        group_labels_grad{end+1}  = lbl;   %#ok<AGROW>

                        % errore basato sulle iterate successive (no xstar)
                        diffs = xseq_i(:,2:end) - xseq_i(:,1:end-1);
                        err_iterdiff = vecnorm(diffs,2,1);
                        err_list_iterdiff{end+1}     = err_iterdiff; %#ok<AGROW>
                        group_labels_iterdiff{end+1} = lbl;          %#ok<AGROW>

                        % errore basato su xstar
                        err_xstar = zeros(1,K);
                        for jj = 1:K
                            err_xstar(jj) = norm(xseq_i(:,jj) - xstar_n);
                        end
                        err_list_xstar{end+1}     = err_xstar; %#ok<AGROW>
                        group_labels_xstar{end+1} = lbl;       %#ok<AGROW>
                    end
                end
            end

            fprintf('--- Grafici AGGREGATI per: %s (tutti i k/type, window 3 e 5) ---\n', base_tag);

            % --- window = 3 ---
            plot_convergence_rate_grouped(err_list_grad, group_labels_grad, ...
                base_tag, 'Broyden31', outdir_rate_grouped3, '||grad f(x_k)||', 3);
            plot_convergence_rate_grouped(err_list_iterdiff, group_labels_iterdiff, ...
                [base_tag '_iterdiff'], 'Broyden31', outdir_rate_grouped_iterdiff3, '||x_k - x_{k-1}||', 3);
            plot_convergence_rate_grouped(err_list_xstar, group_labels_xstar, ...
                [base_tag '_xstar'], 'Broyden31', outdir_rate_grouped_xstar3, '||x_k - x^*||', 3);

            % --- window = 5 ---
            plot_convergence_rate_grouped(err_list_grad, group_labels_grad, ...
                base_tag, 'Broyden31', outdir_rate_grouped5, '||grad f(x_k)||', 5);
            plot_convergence_rate_grouped(err_list_iterdiff, group_labels_iterdiff, ...
                [base_tag '_iterdiff'], 'Broyden31', outdir_rate_grouped_iterdiff5, '||x_k - x_{k-1}||', 5);
            plot_convergence_rate_grouped(err_list_xstar, group_labels_xstar, ...
                [base_tag '_xstar'], 'Broyden31', outdir_rate_grouped_xstar5, '||x_k - x^*||', 5);

            close all;
        end
    end

end
 
disp('Tutti i grafici sono stati generati e salvati in graphs_broyden31/n<N>/<tipo_grafico>/');