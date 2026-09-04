function fig = plot_error_ratio(xseq_list, xstar, method_name, problem_name, figdir, window, min_err)
% PLOT_ERROR_RATIO - per ogni starting point, plotta ad ogni iterazione k:
%   ratio_k = ||x_{k+1} - x*|| / ||x_k - x*||
% Se ratio_k -> 0, la convergenza e' superlineare/quadratica.
% Se ratio_k -> costante c in (0,1), la convergenza e' lineare con
% fattore c.
%
% Sull'asse destro (yyaxis) mostra il rate stimato p_k, calcolato con
% ESTIMATE_RATE_POLYFIT (regressione lineare su finestra scorrevole in
% scala log-log), non piu' con lo stimatore a 3 punti: quest'ultimo
% amplifica il rumore quando i rapporti tra errori consecutivi si
% stabilizzano (divisione tra due log quasi uguali), producendo curve
% del rate che oscillano parecchio senza un motivo reale.
%
% INPUT
%   xseq_list : cell array, uno per starting point, ciascuno n x (k+1)
%   xstar     : minimo di riferimento (n x 1)
%   method_name, problem_name, figdir : per titolo/nome file
%   window    : (opzionale, default 5) ampiezza finestra per il fit del
%               rate. 4-6 e' un buon compromesso; piu' piccola torna
%               rumorosa come lo stimatore a 3 punti, piu' grande smussa
%               troppo eventuali cambi di regime (es. quadratico->lineare)
%   min_err   : (opzionale, default 1e-10) soglia sotto la quale un
%               valore di errore e' trattato come rumore numerico e
%               scartato sia dal ratio sia dal fit del rate

if nargin < 6 || isempty(window),  window  = 5;    end
if nargin < 7 || isempty(min_err), min_err = 1e-10; end

fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.75 0.55]);
colors = lines(numel(xseq_list));
any_plotted = false;

for i = 1:numel(xseq_list)
    xseq = xseq_list{i};
    if iscell(xseq), xseq = xseq{1}; end
    K = size(xseq, 2);
    if K < 2, continue; end

    err = zeros(1, K);
    for k = 1:K
        err(k) = norm(xseq(:,k) - xstar);
    end

    % --- ratio tra errori consecutivi: ||e_{k+1}||/||e_k|| ---
    ratio = nan(1, K-1);
    for k = 1:K-1
        if err(k) > min_err && err(k+1) > min_err
            ratio(k) = err(k+1) / err(k);
        end
    end

    yyaxis left
    plot(1:K-1, ratio, '-o', 'Color', colors(i,:), 'LineWidth', 1.4, ...
         'MarkerSize', 4, 'DisplayName', sprintf('start %d: ratio', i));
    hold on;
    any_plotted = true;

    % --- rate stimato con regressione a finestra scorrevole ---
    if K >= window + 1
        [rate_seq, k_centers] = estimate_rate_polyfit(err, window, min_err);
        if ~isempty(rate_seq)
            yyaxis right
            plot(k_centers, rate_seq, '--s', 'Color', colors(i,:)*0.6, 'LineWidth', 1.0, ...
                 'MarkerSize', 3, 'DisplayName', sprintf('start %d: rate p', i));
            hold on;
        end
    end
end

if ~any_plotted
    warning('Nessuna sequenza valida: grafico saltato');
    close(fig); fig = []; return;
end

yyaxis left
ylabel('||x_{k+1}-x^*|| / ||x_k-x^*||');
xlabel('Iterazione k');

yyaxis right
ylabel(sprintf('Rate estimate p'));
ylim([0 3]);

title(sprintf('%s - %s: consecutive error ratio (sx) and estimate rate(dx)', ...
      method_name, problem_name), 'Interpreter', 'none');
legend('Location', 'bestoutside');
grid on;

if ~isempty(figdir)
    if ~exist(figdir, 'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('errorRatio_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end

end