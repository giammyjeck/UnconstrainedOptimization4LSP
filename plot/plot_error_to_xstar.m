function fig = plot_error_to_xstar(xseq_list, xstar, method_name, problem_name, figdir)
% PLOT_ERROR_TO_XSTAR - grafico CONSIGLIATO: ||x_k - x*|| vs iterazione k,
% scala semilog-y, tutte le starting point nella stessa figura.
%
% Adattata al formato usato in run_experiments_31.m:
%   xseq_list = {results(mask).xseq};   % cell array, uno per starting point
%   xstar_n   = xstarfun(n_target);
%
% INPUT
%   xseq_list  : cell array, uno per starting point, ciascuno n x K
%   xstar      : soluzione di riferimento (n x 1). Se [], il grafico
%                viene saltato con un warning (es. problemi senza
%                soluzione in forma chiusa).
%   method_name, problem_name, figdir : per titolo/nome file
%
% USO TIPICO (dentro run_experiments_31.m, dopo aver costruito xseq_list):
%   plot_error_to_xstar(xseq_list, xstar_n, 'ModifiedNewton', 'Broyden31', 'graphs_broyden31');

fig = [];

if isempty(xstar)
    warning('plot_error_to_xstar: xstar non disponibile per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    return;
end

fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.6 0.45]);
hold on; grid on; box on;
colors = lines(numel(xseq_list));
any_plotted = false;

for i = 1:numel(xseq_list)
    xseq = xseq_list{i};
    if isempty(xseq), continue; end
    K = size(xseq, 2);
    err = zeros(1, K);
    for k = 1:K
        err(k) = norm(xseq(:,k) - xstar);
    end

    semilogy(0:K-1, err, '-o', 'Color', colors(i,:), ...
             'LineWidth', 1.3, 'MarkerSize', 3, ...
             'DisplayName', sprintf('start %d', i));
    any_plotted = true;
end

if ~any_plotted
    warning('plot_error_to_xstar: nessuna traiettoria valida, grafico saltato.');
    close(fig); fig = []; return;
end

xlabel('Iterazione k');
ylabel('||x_k - x^*||');
title(sprintf('%s - %s: error respect to x^*', method_name, problem_name), 'Interpreter','none');
legend('Location','best');

if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('errToXstar_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end

end
