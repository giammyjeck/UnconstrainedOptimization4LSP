function fig = plot_convergence_rate_trig16(xseq_list, method_name, problem_name, figdir, window, min_err, min_logratio)
% PLOT_CONVERGENCE_RATE_TRIG16 - variante di plot_convergence_rate.m
% pensata per trig16, che NON usa xstar (rate su ||x_k - x_{k-1}||,
% eq. 34 delle dispense).
%
% Come plot_convergence_rate.m: i punti scartati perche' il log-rapporto
% e' quasi zero (stagnazione RELATIVA, non rumore macchina) vengono
% marcati con una 'x' alla base del grafico invece di essere nascosti.
%
% INPUT
%   xseq_list    : cell array, uno per starting point, ciascuno n x K
%   method_name, problem_name, figdir : per titolo/nome file
%   window       : (opzionale, default 5) ampiezza mediana mobile su p_k
%   min_err      : (opzionale, default 1e-10) soglia rumore numerico
%   min_logratio : (opzionale, default 1e-3) soglia log-rapporto ~ 0
%
% USO TIPICO:
%   plot_convergence_rate_trig16(xseq_list, 'ModifiedNewton', 'Trig16', 'graphs_trig16');
 
if nargin < 5 || isempty(window),       window       = 5;    end
if nargin < 6 || isempty(min_err),      min_err      = 1e-10; end
if nargin < 7 || isempty(min_logratio), min_logratio = 1e-300; end
 
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
 
    diffs = xseq(:, 2:end) - xseq(:, 1:end-1);
    err   = vecnorm(diffs, 2, 1);
 
    if numel(err) < 3, continue; end
 
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
    warning('plot_convergence_rate_trig16: nessuna sequenza valida per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    close(fig); fig = []; return;
end
 
xlabel('Iterazione k');
ylabel(sprintf('Rate estimate p (||x_k - x_{k-1}||, window %d)', window));
ylim([-2 4]);
title(sprintf('%s - %s: estimate convergence rate (iter diff)', method_name, problem_name), 'Interpreter','none');
legend('Location','best');
 
if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('rate_iterdiff_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end
 
end
 
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
 