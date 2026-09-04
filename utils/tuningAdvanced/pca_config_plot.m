function pca_config_plot(param_matrix, param_names, loss_vals, fig_title)
% PCA_CONFIG_PLOT Visualizza, tramite PCA, come le configurazioni
% (sopravvissute al Refinement 2) si dispongono nello spazio dei
% parametri e come questo si relaziona con la loss. Produce un biplot:
% gli score (proiezione delle configurazioni sulle prime 2 PC) colorati
% per log10(loss), piu' le frecce dei loadings (direzione di ciascun
% parametro originale nello spazio delle componenti principali).
%
% NOTE IMPORTANTI (leggere prima di interpretare il grafico):
% - param_matrix deve gia' contenere i parametri nella scala giusta: per
%   quelli che spaziano piu' ordini di grandezza (c1, beta, kmax, bt,
%   max_cg) va passato log10(valore), NON il valore grezzo, altrimenti
%   la PCA sarebbe dominata da quei parametri solo per via della scala,
%   non della loro reale importanza.
% - Con poche decine di configurazioni sopravvissute al Refinement 2 (e
%   generate da una ricerca a griglia + regole, non campionate a caso),
%   alcuni parametri saranno correlati tra loro nel dataset (es. bt alto
%   e c1 alto tendono a comparire insieme, perche' le regole di
%   escalation li alzano in coppia). La PCA qui mostra percio' "come si
%   muovono insieme i parametri nella ricerca esplorata", non una vera
%   analisi di sensitivita' causale parametro-per-parametro.
%
% param_matrix : matrice (n_configs x n_params), scala gia' preparata
% param_names  : cell array di stringhe (n_params), nomi/etichette assi
% loss_vals    : vettore (n_configs x 1) di loss associate
% fig_title    : titolo della figura
 
    n = size(param_matrix, 1);
    if n < 3
        warning(['pca_config_plot: solo %d configurazioni disponibili, ' ...
            'la PCA e'' poco significativa (servirebbero almeno 4-5 punti).'], n);
    end
 
    col_std = std(param_matrix, 0, 1);
    const_cols = (col_std == 0);
    if any(const_cols)
        fprintf('[PCA] Parametri costanti su tutte le config (varianza nulla, esclusi dalla standardizzazione): %s\n', ...
            strjoin(param_names(const_cols), ', '));
    end
 
    Xz = param_matrix - mean(param_matrix, 1);
    Xz(:, ~const_cols) = Xz(:, ~const_cols) ./ col_std(~const_cols);
    Xz(:, const_cols) = 0;
 
    [coeff, score, ~, ~, explained] = pca(Xz);
 
    if size(score, 2) < 2
        warning('pca_config_plot: meno di 2 componenti principali disponibili, salto il plot.');
        return;
    end
 
    figure('Name', fig_title);
    logloss = log10(max(loss_vals, eps));
    scatter(score(:,1), score(:,2), 90, logloss, 'filled');
    cb = colorbar;
    cb.Label.String = 'log_{10}(loss)';
    hold on;
 
    % Frecce dei loadings, scalate per essere leggibili sulla stessa
    % figura degli score (puramente visivo, non cambia l'interpretazione
    % qualitativa delle direzioni).
    max_score = max(abs(score(:,1:2)), [], 'all');
    max_coeff = max(abs(coeff(:,1:2)), [], 'all');
    if max_coeff > 0
        scale = 0.8 * max_score / max_coeff;
    else
        scale = 1;
    end
    for j = 1:size(coeff, 1)
        quiver(0, 0, coeff(j,1)*scale, coeff(j,2)*scale, 0, ...
            'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.6);
        text(coeff(j,1)*scale*1.12, coeff(j,2)*scale*1.12, param_names{j}, ...
            'FontWeight', 'bold', 'FontSize', 9);
    end
    hold off;
 
    xlabel(sprintf('PC1 (%.1f%% varianza)', explained(1)));
    ylabel(sprintf('PC2 (%.1f%% varianza)', explained(2)));
    title(fig_title);
    grid on;
 
    fprintf('\n[PCA] "%s": varianza spiegata PC1=%.1f%%, PC2=%.1f%%, cumulata=%.1f%%\n', ...
        fig_title, explained(1), explained(2), sum(explained(1:min(2,numel(explained)))));
end
 