function fig = plot_convergence_rate_grouped(err_list, group_labels, method_name, problem_name, figdir, err_label, window, min_err, min_logratio)
% PLOT_CONVERGENCE_RATE_GROUPED - aggrega in un UNICO grafico le curve
% di rate di piu' combinazioni (es. tutti i k={4,8,12} x type={h,hi} di
% case1), assegnando lo STESSO colore a tutte le run che appartengono
% allo stesso gruppo (stesso k, stesso type) e un colore DIVERSO ad ogni
% gruppo. E' la versione "aggregata" di plot_convergence_rate.m /
% plot_convergence_rate_trig16.m / plot_convergence_rate_xstar.m, che
% invece producono un grafico per singola combinazione.
%
% Non ricalcola nulla sulla natura dell'errore (gradiente, iterate
% successive, o distanza da xstar): riceve gia' le sequenze di errore
% pronte, cosi' la stessa funzione serve per tutte e tre le varianti -
% e' il chiamante a decidere COSA passare come err_list.
%
% INPUT
%   err_list     : cell array, un vettore [1 x K_i] di errori PER OGNI
%                  RUN (non per combinazione: se un gruppo k4_h ha 4
%                  starting point convergenti, contribuisce 4 elementi
%                  a err_list, tutti con la stessa etichetta di gruppo)
%   group_labels : cell array di stringhe, stessa lunghezza di err_list,
%                  un'etichetta di gruppo per ogni run (es. 'k4_h',
%                  'k8_hi', ...). Le run con la stessa etichetta
%                  condividono colore e comparivano una sola volta in
%                  legenda.
%   method_name, problem_name, figdir : per titolo/nome file
%   err_label    : descrizione dell'errore per l'asse y (es.
%                  '||grad f(x_k)||', '||x_k - x*||', '||x_k - x_{k-1}||')
%   window       : (opzionale, default 5) ampiezza mediana mobile su p_k
%   min_err      : (opzionale, default 1e-10) soglia rumore numerico
%   min_logratio : (opzionale, default 1e-3) soglia log-rapporto ~ 0
%
% OUTPUT
%   fig : handle della figura (vuoto se nessuna run valida)
%
% USO TIPICO (dentro run_experiments_31.m, per aggregare case1):
%
%   err_list = {}; group_labels = {};
%   for kk = k_list
%       for type = fdtypes
%           mask = ... & [results.k]==kk & [results.type]==type & [results.success];
%           xs = {results(mask).xseq};
%           typestr = ternary(type==1,'h','hi');
%           lbl = sprintf('k%d_%s', kk, typestr);
%           for i = 1:numel(xs)
%               gnorm = arrayfun(@(j) norm(grad_exact(xs{i}(:,j))), 1:size(xs{i},2));
%               err_list{end+1}   = gnorm;      %#ok<AGROW>
%               group_labels{end+1} = lbl;      %#ok<AGROW>
%           end
%       end
%   end
%   plot_convergence_rate_grouped(err_list, group_labels, ...
%       'ModifiedNewton_case1', 'Broyden31', outdir, '||grad f(x_k)||');
 
if nargin < 7 || isempty(window),       window       = 5;    end
if nargin < 8 || isempty(min_err),      min_err      = 1e-10; end
if nargin < 9 || isempty(min_logratio), min_logratio = 1e-500; end
 
fig = [];
 
if isempty(err_list)
    warning('plot_convergence_rate_grouped: nessuna run per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    return;
end
 
% --- un colore per ogni gruppo UNICO (non per run) ---
[unique_groups, ~, group_idx] = unique(group_labels, 'stable');
n_groups = numel(unique_groups);
colors = lines(n_groups);
 
fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.75 0.55]);
hold on; grid on; box on;
 
any_plotted   = false;
legend_drawn  = false(1, n_groups);   % una voce in legenda per gruppo (rate)
stagn_drawn   = false;                % una voce in legenda totale per le 'x' di stagnazione
 
for r = 1:numel(err_list)
    err = err_list{r};
    if isempty(err) || numel(err) < 3, continue; end
 
    g = group_idx(r);
    col = colors(g,:);
 
    [rate_seq, k_idx, ~, flag_stagn] = estimate_rate_seq(err, window, min_err, min_logratio);
    if isempty(rate_seq) || all(isnan(rate_seq)), continue; end
 
    if ~legend_drawn(g)
        dname = unique_groups{g};
        legend_drawn(g) = true;
    else
        dname = '';
    end
 
    plot(k_idx, rate_seq, '-o', 'Color', col, ...
         'LineWidth', 1.1, 'MarkerSize', 3, ...
         'DisplayName', dname, ...
         'HandleVisibility', ternary_local(isempty(dname), 'off', 'on'));
    any_plotted = true;
 
    if any(flag_stagn)
        if ~stagn_drawn
            sdname = 'scartato: log-ratio \approx 0 (stagnazione)';
            stagn_drawn = true;
        else
            sdname = '';
        end
        plot(k_idx(flag_stagn), zeros(1, nnz(flag_stagn)), 'x', ...
             'Color', col*0.5 + 0.5*[0.5 0.5 0.5], ...
             'MarkerSize', 5, 'LineWidth', 1.0, ...
             'DisplayName', sdname, ...
             'HandleVisibility', ternary_local(isempty(sdname), 'off', 'on'));
    end
end
 
if ~any_plotted
    warning('plot_convergence_rate_grouped: nessuna run valida per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    close(fig); fig = []; return;
end
 
xlabel('Iterazione k');
ylabel(sprintf('Rate estimate p (%s, window %d)', err_label, window));
ylim([-0.3 4]);
title(sprintf('%s - %s: Estimate rate (color = k/type)', method_name, problem_name), 'Interpreter','none');
legend('Location','bestoutside');
 
if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('rate_grouped_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end
 
end
 
function out = ternary_local(cond, a, b)
    if cond, out = a; else, out = b; end
end
 