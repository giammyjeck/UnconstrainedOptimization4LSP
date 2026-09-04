function fig = plot_contour_paths(f, xseq_list, method_name, problem_name, figdir)
% PLOT_CONTOUR_PATHS - grafico OBBLIGATORIO (Sez. 2.1 assignment):
% "top view of the function and of all the sequence paths, for n=2.
%  Collect in the same figure the paths for each different starting point"
%
% Adattata al formato usato in run_experiments_31.m:
%   xseq_list = {results(mask).xseq};   % cell array, uno per starting point
%
% INPUT
%   f            : function handle f(x), x in R^2, per le curve di livello
%   xseq_list    : cell array, uno per starting point, ciascuno 2 x K
%                  (K = numero di iterazioni +1, x0 incluso)
%   method_name, problem_name : per titolo/nome file (es. 'ModifiedNewton', 'Broyden31')
%   figdir       : cartella di output (creata se non esiste, '' per non salvare)
%
% USO TIPICO (dentro run_experiments_31.m, dopo aver costruito xseq_list):
%   plot_contour_paths(f, xseq_list, 'ModifiedNewton', 'Broyden31', 'graphs_broyden31');

% --- controllo n=2 ---
n_check = size(xseq_list{1}, 1);
assert(n_check == 2, 'plot_contour_paths richiede n=2 (trovato n=%d).', n_check);

%% --- range degli assi: unione di tutte le traiettorie con margine ---
allX = [];
for i = 1:numel(xseq_list)
    xs = xseq_list{i};
    if isempty(xs), continue; end
    allX = [allX, xs]; %#ok<AGROW>
end
if isempty(allX)
    warning('plot_contour_paths: nessuna traiettoria valida, grafico saltato.');
    fig = []; return;
end

xmin = min(allX(1,:)); xmax = max(allX(1,:));
ymin = min(allX(2,:)); ymax = max(allX(2,:));
mx = 0.15*(xmax-xmin+eps); my = 0.15*(ymax-ymin+eps);
xr = linspace(xmin-mx, xmax+mx, 300);
yr = linspace(ymin-my, ymax+my, 300);
[X, Y] = meshgrid(xr, yr);
Z = zeros(size(X));
for i = 1:numel(xr)
    for j = 1:numel(yr)
        Z(j,i) = f([X(j,i); Y(j,i)]);
    end
end

%% --- figura ---
fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.6 0.6]);
hold on; grid on; box on;

% curve di livello (log-spaced per leggibilita' se Z ha grande range)
zpos = Z(Z>0);
if ~isempty(zpos)
    levels = logspace(log10(max(min(zpos),1e-6)), log10(max(zpos)), 25);
    contour(X, Y, Z, levels, 'LineColor', [0.6 0.6 0.6]);
else
    contour(X, Y, Z, 25, 'LineColor', [0.6 0.6 0.6]);
end
colormap(parula);

colors = lines(numel(xseq_list));
for i = 1:numel(xseq_list)
    xs = xseq_list{i};
    if isempty(xs), continue; end
    plot(xs(1,:), xs(2,:), '-o', 'Color', colors(i,:), ...
         'LineWidth', 1.3, 'MarkerSize', 3, ...
         'DisplayName', sprintf('start %d', i));
    plot(xs(1,1), xs(2,1), 'k^', 'MarkerFaceColor', colors(i,:), 'MarkerSize', 8, 'HandleVisibility','off');
    plot(xs(1,end), xs(2,end), 'ks', 'MarkerFaceColor', colors(i,:), 'MarkerSize', 8, 'HandleVisibility','off');
end

xlabel('x_1'); ylabel('x_2');
title(sprintf('%s - %s: trajectories (n=2) - triangle=start, square=finale', ...
      method_name, problem_name), 'Interpreter','none');
legend('Location','bestoutside');

%% --- salvataggio ---
if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('contour_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end

end
