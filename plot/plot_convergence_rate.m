function fig = plot_convergence_rate(xseq_list, gradf, method_name, problem_name, figdir, window, min_err, min_logratio)
% PLOT_CONVERGENCE_RATE - grafico OBBLIGATORIO (Sez. 2.1 assignment):
% "Experimental rates of convergence for the sequences that converged."
%
% Il rate p_k e' stimato con ESTIMATE_RATE_SEQ.m (formula puntuale a 3
% punti, eq. 33, aggregata con mediana mobile). I punti scartati perche'
% |log-rapporto| ~ 0 (due errori consecutivi quasi identici IN RAPPORTO,
% non per rumore macchina) vengono marcati esplicitamente con una 'x'
% grigia alla base del grafico (y=0): NON sono nascosti, cosi' se
% compaiono in massa verso la fine di una sequenza e' un segnale da
% controllare sull'algoritmo (backtracking pesante, tau_k che cresce,
% kmax raggiunto), non un artefatto dello stimatore.
%
% INPUT
%   xseq_list    : cell array, uno per starting point, ciascuno n x K
%   gradf        : function handle per il gradiente (es. grad_exact)
%   method_name, problem_name, figdir : per titolo/nome file
%   window       : (opzionale, default 5) ampiezza mediana mobile su p_k
%   min_err      : (opzionale, default 1e-10) soglia rumore numerico
%   min_logratio : (opzionale, default 1e-3) soglia log-rapporto ~ 0
%
% USO TIPICO:
%   plot_convergence_rate(xseq_list, grad_exact, 'ModifiedNewton', 'Broyden31', 'graphs_broyden31');
 
if nargin < 6 || isempty(window),       window       = 5;    end
if nargin < 7 || isempty(min_err),      min_err      = 1e-10; end
if nargin < 8 || isempty(min_logratio), min_logratio = 1e-500; end
 
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
 
    gnorm = zeros(1, K);
    for k = 1:K
        gnorm(k) = norm(gradf(xseq(:,k)));
    end
 
    [rate_seq, k_idx, ~, flag_stagn] = estimate_rate_seq(gnorm, window, min_err, min_logratio);
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
            % niente HandleVisibility per non duplicare la legenda,
            % ma il colore per starting point aiuta comunque a capire
            % quale sequenza sta stagnando
        end
        plot(k_idx(flag_stagn), zeros(1, nnz(flag_stagn)), 'x', ...
             'Color', colors(i,:)*0.5 + 0.5*[0.5 0.5 0.5], ...
             'MarkerSize', 6, 'LineWidth', 1.2, ...
             'DisplayName', dname, ...
             'HandleVisibility', ternary(isempty(dname), 'off', 'on'));
    end
end
 
if ~any_plotted
    warning('plot_convergence_rate: nessuna sequenza valida per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    close(fig); fig = []; return;
end
 
xlabel('Iterazione k');
ylabel(sprintf('Rate estimate p (gradfk, window %d)', window));
ylim([-0.3 4]);   % un filo sotto 0 per distinguere le 'x' (a y=0) dalla curva
title(sprintf('%s - %s: estimate convergence rate', method_name, problem_name), 'Interpreter','none');
legend('Location','best');
 
if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('rate_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end
 
end
 
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
 