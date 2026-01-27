function outputs(experimentalMatrix, f, gradf)
% PLOT_OPTIMIZATION_RESULTS Genera grafici di analisi per il Truncated Newton
%
% INPUTS:
%   experimentalMatrix: Struct contenente i risultati delle run
%   f:                  Function handle della funzione obiettivo
%   gradf:              Function handle del gradiente

    % Creating folder for saving outputs
    outdir = "out_modified_exact";
    figdir = fullfile(outdir,"figures");
    tabdir = fullfile(outdir,"tables");
    
    if ~exist(outdir,"dir"), mkdir(outdir); end
    if ~exist(figdir,"dir"), mkdir(figdir); end
    if ~exist(tabdir,"dir"), mkdir(tabdir); end

    dims = [experimentalMatrix.n];
    num_dims = length(dims);
    
    
    % Calcola medie solo sulle run di successo (o tutte se preferisci)
    avg_times = zeros(1, num_dims);
    avg_iters = zeros(1, num_dims);
    
    for i = 1:num_dims
        runs = experimentalMatrix(i).runs;
        times = [runs.time];
        iters = [runs.k];
        
        % Media semplice (puoi filtrare per successi se vuoi essere rigoroso)
        avg_times(i) = mean(times);
        avg_iters(i) = mean(iters);
    end
    
    figure('Name', 'Scalability Analysis', 'Color', 'w');
    
    % Subplot 1: Tempo vs Dimensione
    subplot(1, 2, 1);
    loglog(dims, avg_times, '-ro', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
    grid on;
    xlabel('Dimensione (n)', 'FontSize', 12);
    ylabel('Tempo Medio (s)', 'FontSize', 12);
    title('Scalabilità Temporale', 'FontSize', 14);
    axis tight; 
    
    % Subplot 2: Iterazioni vs Dimensione
    subplot(1, 2, 2);
    semilogx(dims, avg_iters, '-bs', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    grid on;
    xlabel('Dimensione (n)', 'FontSize', 12);
    ylabel('Iterazioni Medie', 'FontSize', 12);
    title('Robustezza Algoritmica', 'FontSize', 14);
    
    sgtitle('Analisi Scalabilità Truncated Newton');
    
    %% 2. CONVERGENCE ANALYSIS (Dimensione più alta)
    % Mostra l'andamento del gradiente per la dimensione massima testata
    [max_n, idx_max] = min(dims);
    runs_max = experimentalMatrix(idx_max).runs;
    num_runs = length(runs_max);
    fixed_colors = lines(num_runs); 
    
    figure('Name', sprintf('Convergence n=%d', max_n), 'Color', 'w');
    hold on;
    
    for s = 1:length(runs_max)
        % Ricostruisce la storia del gradiente usando xseq salvato
        xseq = runs_max(s).xseq;
        gnorm_history = zeros(1, size(xseq, 2));
        for k = 1:size(xseq, 2)
            gnorm_history(k) = norm(gradf(xseq(:, k)));
        end
        
        semilogy(0:length(gnorm_history)-1, gnorm_history, '.-', ...
            'Color', fixed_colors(s,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Start Pt %d', s));
    end
    
    grid on;
    xlabel('Iterazioni (k)', 'FontSize', 12);
    ylabel('||\nabla f(x_k)||', 'FontSize', 12);
    title(sprintf('Convergenza del Gradiente (n = %d)', max_n), 'FontSize', 14);
    legend('show', 'Location', 'northeast');
    hold off;
    
    %% 3. 2D TRAJECTORY PLOT (Solo se n=2 è presente)
    idx_2d = find(dims == 2, 1);
    
    if ~isempty(idx_2d)
        runs_2d = experimentalMatrix(idx_2d).runs;
        
        figure('Name', '2D Trajectory', 'Color', 'w');
        hold on;
        
        % Determina i limiti del grafico basandosi su tutte le traiettorie
        all_x = []; all_y = [];
        for s = 1:length(runs_2d)
            all_x = [all_x, runs_2d(s).xseq(1,:)]; %#ok<AGROW>
            all_y = [all_y, runs_2d(s).xseq(2,:)]; %#ok<AGROW>
        end
        
        padding = 1;
        x_min = min(all_x) - padding; x_max = max(all_x) + padding;
        y_min = min(all_y) - padding; y_max = max(all_y) + padding;
        
        % Genera griglia per contour
        [X, Y] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
        Z = zeros(size(X));
        for i = 1:size(X, 1)
            for j = 1:size(X, 2)
                Z(i,j) = f([X(i,j); Y(i,j)]);
            end
        end
        
        % Plot Curve di Livello
        contour(X, Y, Z, 50, 'LineWidth', 0.5, 'ShowText', 'off');
        colormap parula; % O 'jet', 'turbo'
        
        % Plot Traiettorie
        for s = 1:length(runs_2d)
            xseq = runs_2d(s).xseq;
            plot(xseq(1,:), xseq(2,:), '-o', ...
                'Color', fixed_colors(s,:), ... 
                'LineWidth', 1.5, 'MarkerSize', 4, ...
                'DisplayName', sprintf('Run %d', s));
            % Evidenzia punto iniziale e finale
            plot(xseq(1,1), xseq(2,1), 'g*', 'MarkerSize', 8, 'HandleVisibility', 'off'); % Start
            plot(xseq(1,end), xseq(2,end), 'rx', 'MarkerSize', 8, 'HandleVisibility', 'off'); % End
        end
        
        xlabel('x_1'); ylabel('x_2');
        title('Traiettorie di Ottimizzazione (n = 2)');
        legend('show');
        grid on;
        axis([x_min x_max y_min y_max]);
        hold off;
    else
        fprintf('\n[Info] Nessun grafico 2D generato (dimensione n=2 non trovata nei test).\n');
    end

end