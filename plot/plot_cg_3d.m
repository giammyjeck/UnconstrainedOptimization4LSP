function fig = plot_cg_3d(f, iter_trace, k_idx, method_name, problem_name, figdir)
% PLOT_CG_3D - superficie 3D di f(x) attorno a x_k, con il percorso delle
% iterate parziali del CG (z_0=0, z_1, z_2, ..., p_k) disegnato SOPRA la
% superficie (usando il valore reale di f in ogni punto), per vedere
% fisicamente come il CG "scende" verso il minimo lungo la direzione di
% Newton, passo dopo passo.
%
% INPUT
%   f            : function handle f(x), x in R^2
%   iter_trace   : trace(k_idx) da truncated_newton_method_diagnostic.m
%   k_idx        : indice iterazione esterna (per titolo/nome file)
%   method_name, problem_name, figdir : per titolo/nome file

xb = iter_trace.x_before;
assert(numel(xb) == 2, 'plot_cg_3d richiede n=2.');
cgt = iter_trace.cg_trace;
nJ = numel(cgt);

if nJ == 0
    Zpts = zeros(2,1);
else
    Zpts = [zeros(2,1), reshape([cgt.z], 2, nJ)];
end
pts = xb + Zpts;               % 2 x (nJ+1), punti nel piano (x1,x2)
fvals = zeros(1, size(pts,2));
for i = 1:size(pts,2)
    fvals(i) = f(pts(:,i));
end

% --- griglia per la superficie, centrata sull'inviluppo delle iterate ---
mx = 0.35*(max(pts(1,:))-min(pts(1,:))+0.5);
my = 0.35*(max(pts(2,:))-min(pts(2,:))+0.5);
xr = linspace(min(pts(1,:))-mx, max(pts(1,:))+mx, 100);
yr = linspace(min(pts(2,:))-my, max(pts(2,:))+my, 100);
[X,Y] = meshgrid(xr,yr);
Z = zeros(size(X));
for i=1:numel(xr)
    for jj=1:numel(yr)
        Z(jj,i) = f([X(jj,i); Y(jj,i)]);
    end
end

fig = figure('Color','w','Units','normalized','Position',[0.08 0.08 0.7 0.65]);
surf(X, Y, Z, 'FaceAlpha', 0.55, 'EdgeColor', 'none');
colormap(parula); hold on; grid on; box on;
% curve di livello proiettate alla base, per riferimento
zbase = min(Z(:)) - 0.05*(max(Z(:))-min(Z(:))+eps);
contour3(X, Y, Z + zbase - min(Z(:)), 20, 'LineColor', [0.6 0.6 0.6]);

% --- percorso delle iterate CG, "appoggiato" sulla superficie ---
cmapJ = winter(max(size(pts,2)-1,1));
plot3(pts(1,1), pts(2,1), fvals(1), '^', 'MarkerFaceColor','g', 'MarkerEdgeColor','k', ...
      'MarkerSize', 11, 'DisplayName', 'x_k (z_0=0)');
for j = 2:size(pts,2)
    plot3(pts(1,j-1:j), pts(2,j-1:j), fvals(j-1:j), '-', 'Color', cmapJ(j-1,:), ...
          'LineWidth', 2.5, 'HandleVisibility','off');
    plot3(pts(1,j), pts(2,j), fvals(j), 'o', 'Color', cmapJ(j-1,:), ...
          'MarkerFaceColor', cmapJ(j-1,:), 'MarkerSize', 7, ...
          'DisplayName', sprintf('z_%d', j-1));
end

xlabel('x_1'); ylabel('x_2'); zlabel('f(x)');
title(sprintf('%s - %s: iter %d, percorso CG sulla superficie di f (uscita: %s)', ...
      method_name, problem_name, k_idx, iter_trace.cg_exit), ...
      'FontSize', 10, 'Interpreter', 'none');
legend('Location','best');
view(-35, 30);

if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('cg3d_%s_%s_iter%02d', method_name, problem_name, k_idx));
    exportgraphics(fig, fullfile(figdir, [fname '.png']), 'Resolution', 200);
end

end
