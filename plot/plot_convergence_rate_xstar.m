function fig = plot_convergence_rate_xstar(xseq_list, xstar, method_name, problem_name, figdir, window, min_err, min_logratio)
% PLOT_CONVERGENCE_RATE_XSTAR - variante di plot_convergence_rate.m /
% plot_convergence_rate_trig16.m che usa ||x_k - x*|| come errore, con
% lo stesso stimatore ESTIMATE_RATE_SEQ.m (formula puntuale a 3 punti,
% eq. 33, aggregata con mediana mobile robusta) invece di
% ESTIMATE_RATE_POLYFIT.m usato in plot_error_ratio.m.
%
% E' la terza variante della stessa famiglia:
%   - plot_convergence_rate.m         -> err = ||grad f(x_k)||
%   - plot_convergence_rate_trig16.m  -> err = ||x_k - x_{k-1}||  (no xstar)
%   - plot_convergence_rate_xstar.m   -> err = ||x_k - x*||       (QUESTA)
%
% Ha senso solo per problemi con un x* di riferimento noto e univoco
% (es. Broyden31): su problemi con minimi multipli equivalenti (es.
% Trig16, periodica) la distanza da un singolo x* non riflette la vera
% velocita' di convergenza - per quel caso usare
% plot_convergence_rate_trig16.m, che non richiede xstar.
%
% Come plot_convergence_rate.m: i punti scartati perche' il log-rapporto
% e' quasi zero (stagnazione RELATIVA, non rumore macchina) vengono
% marcati con una 'x' alla base del grafico invece di essere nascosti.
%
% INPUT
%   xseq_list    : cell array, uno per starting point, ciascuno n x K
%   xstar        : soluzione di riferimento (n x 1)
%   method_name, problem_name, figdir : per titolo/nome file
%   window       : (opzionale, default 5) ampiezza mediana mobile su p_k
%   min_err      : (opzionale, default 1e-10) soglia rumore numerico
%   min_logratio : (opzionale, default 1e-3) soglia log-rapporto ~ 0
%
% USO TIPICO:
%   plot_convergence_rate_xstar(xseq_list, xstar_n, tag, 'Broyden31', outdir_rate_xstar);
 
if nargin < 6 || isempty(window),       window       = 5;    end
if nargin < 7 || isempty(min_err),      min_err      = 1e-10; end
if nargin < 8 || isempty(min_logratio), min_logratio = 1e-500; end
 
fig = [];
 
if isempty(xstar)
    warning('plot_convergence_rate_xstar: xstar non disponibile per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    return;
end
 
fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.65 0.5]);
hold on; grid on; box on;
colors = lines(numel(xseq_list));
any_plotted = false;
any_stagn_legend = false;
 
for i = 1:numel(xseq_list)
    xseq = xseq_list{i};
    if isempty(xseq), continue; end
    K = size(xseq, 2);
    if K < 3, continue; end
 
    err = zeros(1, K);
    for k = 1:K
        err(k) = norm(xseq(:,k) - xstar);
    end
 
    [rate_seq, k_idx, ~, flag_stagn] = estimate_rate_seq(err, window, min_err, min_logratio);
    if isempty(rate_seq) || all(isnan(rate_seq)), continue; end
 
    plot(k_idx, rate_seq, '-o', 'Color', colors(i,:), ...
         'LineWidth', 1.3, 'MarkerSize', 3, ...
         'DisplayName', sprintf('start %d', i));
    any_plotted = true;
 
    if any(flag_stagn)
        if ~any_stagn_legend
            dname = 'scartato: log-ratio \approx 0 (stagnazione)';
            any_stagn_legend = true;
        else
            dname = '';
        end
        plot(k_idx(flag_stagn), zeros(1, nnz(flag_stagn)), 'x', ...
             'Color', colors(i,:)*0.5 + 0.5*[0.5 0.5 0.5], ...
             'MarkerSize', 6, 'LineWidth', 1.2, ...
             'DisplayName', dname, ...
             'HandleVisibility', ternary(isempty(dname), 'off', 'on'));
    end
end
 
if ~any_plotted
    warning('plot_convergence_rate_xstar: nessuna sequenza valida per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    close(fig); fig = []; return;
end
 
xlabel('Iterazione k');
ylabel(sprintf('Rate estimate p (||x_k - x^*||, window %d)', window));
ylim([-0.3 4]);
title(sprintf('%s - %s: estimate convergence rate (xstar)', method_name, problem_name), 'Interpreter','none');
legend('Location','best');
 
if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('rate_xstar_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end
 
end
 
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
 