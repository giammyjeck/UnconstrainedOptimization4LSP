function outputs(experimentalMatrix, f, gradf, pname)
% OUTPUTS - Genera grafici (scalabilità, traiettorie, rates) e tabelle (CSV)
%
% INPUTS:
%   experimentalMatrix: Struct con i risultati
%   f, gradf:           Handle funzione e gradiente
%   pname:              Stringa col nome del problema (es. "Problem16_Trig")

    % 1. Setup Cartelle
    outdir = "outputs";
    figdir = fullfile(outdir,"figures");
    tabdir = fullfile(outdir,"tables");
    
    if ~exist(outdir,"dir"), mkdir(outdir); end
    if ~exist(figdir,"dir"), mkdir(figdir); end
    if ~exist(tabdir,"dir"), mkdir(tabdir); end

    dims = [experimentalMatrix.n];
    num_dims = length(dims);
    allRows = {}; % Per la tabella globale
    kmax = 1000;  % Assunto dal main

    %% --- LOOP PER DIMENSIONE (Tabelle e Rates) ---
    for i = 1:num_dims
        n = dims(i);
        runs = experimentalMatrix(i).runs;
        num_runs = length(runs);
        
        % Inizializzazione dati per tabella singola dimensione
        tableRows = cell(num_runs, 6);
        
        % --- Grafico Rates (log10 ||grad|| vs k) ---
        figR = figure('Visible', 'off', 'Name', sprintf('Rates %s n=%d', pname, n));
        hold on; grid on;
        success_indices = [];

        for s = 1:num_runs
            xseq = runs(s).xseq;
            gn = runs(s).gradfk_norm;
            k = runs(s).k;
            success = gn < 1e-6; % tolgrad
            
            % Ricostruzione sequenza norma gradiente per il grafico
            gnorm_history = zeros(1, size(xseq, 2));
            for it = 1:size(xseq, 2)
                gnorm_history(it) = norm(gradf(xseq(:, it)));
            end
            
            % Calcolo rate sperimentale (usando la funzione helper sotto)
            rate_exp = estimate_order_p(gnorm_history);
            
            % Accumulo dati per tabelle
            startID = "rand" + (s-1); if s==1, startID = "xbar"; end
            iters_str = sprintf("%d/%d", k, kmax);
            succ_str = "no"; if success, succ_str = "yes"; end
            
            tableRows(s, :) = {char(startID), gn, iters_str, char(succ_str), rate_exp, runs(s).time};
            allRows(end+1, :) = {char(pname), n, char(startID), gn, iters_str, char(succ_str), rate_exp, runs(s).time}; %#ok<AGROW>

            if success
                plot(0:length(gnorm_history)-1, log10(gnorm_history + 1e-20), 'LineWidth', 1.2);
                success_indices = [success_indices, s]; %#ok<AGROW>
            end
        end
        
        % Salvataggio Grafico Rates
        xlabel('Iterazione k'); ylabel('log_{10}(||\nabla f(x_k)||)');
        title(sprintf('%s (n=%d) - Convergence Rates', pname, n));
        exportgraphics(figR, fullfile(figdir, sprintf("%s_n%d_rates.png", pname, n)));
        close(figR);

        % Creazione e salvataggio Tabella CSV per n
        Tn = cell2table(tableRows, 'VariableNames', {'start_pt_ID','grad_norm','iters_over_max','success_flag','rate_conv_exp','time_s'});
        % Aggiunta riga Average
        if ~isempty(success_indices)
            avg_gn = mean([tableRows{success_indices, 2}]);
            avg_rate = mean([tableRows{success_indices, 5}], 'omitnan');
            avg_time = mean([tableRows{success_indices, 6}]);
            avgRow = {"Avg(successes)", avg_gn, "", "-", avg_rate, avg_time};
            Tn = [Tn; cell2table(avgRow, 'VariableNames', Tn.Properties.VariableNames)];
        end
        writetable(Tn, fullfile(tabdir, sprintf("%s_n%d_table.csv", pname, n)));
    end

    %% --- SCALABILITY ANALYSIS ---
    avg_times = arrayfun(@(d) mean([d.runs.time]), experimentalMatrix);
    avg_iters = arrayfun(@(d) mean([d.runs.k]), experimentalMatrix);
    
    figS = figure('Name', 'Scalability', 'Color', 'w', 'Visible', 'on');
    subplot(1, 2, 1);
    loglog(dims, avg_times, '-ro', 'LineWidth', 2); grid on;
    xlabel('n'); ylabel('Avg Time (s)'); title('Scalabilità Tempo');
    subplot(1, 2, 2);
    semilogx(dims, avg_iters, '-bs', 'LineWidth', 2); grid on;
    xlabel('n'); ylabel('Avg Iters'); title('Robustezza Iterazioni');
    exportgraphics(figS, fullfile(figdir, sprintf("%s_scalability.png", pname)));

    %% --- 2D TRAJECTORY PLOT (Full + Zoom) ---
    idx_2d = find(dims == 2, 1);
    if ~isempty(idx_2d)
        runs_2d = experimentalMatrix(idx_2d).runs;
        figP = figure('Name', '2D Paths', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.4]);
        
        % Raccolta punti per limiti
        all_pts = [];
        for s=1:length(runs_2d), all_pts = [all_pts, runs_2d(s).xseq]; end %#ok<AGROW>
        
        % Subplot 1: Full View | Subplot 2: Zoom (95% quantile)
        lims = { [min(all_pts(1,:))-1, max(all_pts(1,:))+1, min(all_pts(2,:))-1, max(all_pts(2,:))+1], ...
                 [quantile(all_pts(1,:), 0.05)-0.5, quantile(all_pts(1,:), 0.95)+0.5, ...
                  quantile(all_pts(2,:), 0.05)-0.5, quantile(all_pts(2,:), 0.95)+0.5] };
        titles = {"Full View", "Robust Zoom"};

        for sub = 1:2
            ax = subplot(1, 2, sub); hold on;
            % Disegno contour logaritmico
            x1 = linspace(lims{sub}(1), lims{sub}(2), 100);
            x2 = linspace(lims{sub}(3), lims{sub}(4), 100);
            [X1, X2] = meshgrid(x1, x2); Z = zeros(size(X1));
            for j=1:numel(X1), Z(j) = f([X1(j); X2(j)]); end
            contour(ax, X1, X2, log10(Z - min(Z(:)) + 1e-9), 25, 'LineColor', [0.7 0.7 0.7]);
            
            % Plot percorsi
            colors = lines(length(runs_2d));
            for s = 1:length(runs_2d)
                plot(ax, runs_2d(s).xseq(1,:), runs_2d(s).xseq(2,:), '-o', 'Color', colors(s,:), 'MarkerSize', 3);
            end
            title(titles{sub}); grid on; axis(lims{sub});
        end
        sgtitle(sprintf('Traiettorie %s (n=2)', pname));
        exportgraphics(figP, fullfile(figdir, sprintf("%s_n2_paths.png", pname)));
    end

    % Salvataggio Tabella Globale
    Tall = cell2table(allRows, 'VariableNames', {'problem','n','start_pt_ID','grad_norm','iters_over_max','success_flag','rate_conv_exp','time_s'});
    writetable(Tall, fullfile(tabdir, sprintf("%s_GLOBAL_table.csv", pname)));
end

%% --- HELPER: Stima ordine di convergenza p ---
function p_est = estimate_order_p(gnseq)
    gn = gnseq(gnseq > 1e-12); % Filtra rumore numerico
    if numel(gn) < 5, p_est = NaN; return; end
    
    % Prende gli ultimi 5-10 valori prima della convergenza
    vals = gn(max(1, end-7):end);
    ps = [];
    for i = 2:numel(vals)-1
        e_k_minus_1 = vals(i-1);
        e_k = vals(i);
        e_k_plus_1 = vals(i+1);
        
        p = log(e_k_plus_1 / e_k) / log(e_k / e_k_minus_1);
        if p > 0 && p < 4, ps = [ps, p]; end %#ok<AGROW>
    end
    if isempty(ps), p_est = NaN; else, p_est = median(ps); end
end