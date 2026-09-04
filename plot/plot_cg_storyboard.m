function figs = plot_cg_storyboard(f, gradf, iter_trace, k_idx, method_name, problem_name, figdir)
% PLOT_CG_STORYBOARD - segue l'iter decisionale dell'algoritmo passo per
% passo, UN PANNELLO PER OGNI DECISIONE, per una singola iterazione
% esterna k:
%
%   Pannello 0: si fissa d_0 = -grad f(x_k) (steepest descent): e' SEMPRE
%               la direzione con cui parte il CG, mostrata da sola.
%
%   Pannello j (j=1..cg_iters): per lo step CG j-esimo:
%       - direzione d_{j} testata (la conjugate direction corrente)
%       - TEST DI CURVATURA: d_j' H d_j
%           * se <= soglia -> negativa/nulla: ci si FERMA e si tiene
%             l'ultima direzione buona (z_{j-1}, o -grad se j==1)
%           * se > soglia -> si accetta il passo, si calcola la nuova
%             iterata z_j = z_{j-1} + alpha_j*d_j, e si confronta
%             ||r_j|| (residuo del sistema Newton) con la tolleranza eta
%             per decidere se continuare o fermarsi
%
%   Pannello finale: direzione p_k accettata, poi CONDIZIONE DI ARMIJO
%       mostrata come curva REALE phi(alpha) = f(x_k + alpha*p_k) per
%       alpha continuo, confrontata con la retta soglia
%       f(x_k) + c1*alpha*(grad'*p_k); i punti effettivamente provati dal
%       backtracking sono marcati sopra la curva.
%
% Ogni pannello e' salvato come figura SEPARATA (piu' leggibile di un
% unico subplot fitto), numerata in ordine, cosi' si possono scorrere
% in sequenza come uno storyboard.
%
% INPUT
%   f, gradf     : function handle (n=2)
%   iter_trace   : trace(k_idx) prodotto da
%                  truncated_newton_method_diagnostic.m
%   k_idx        : indice iterazione esterna (per titolo/nome file)
%   method_name, problem_name, figdir : per titolo/nome file
%
% OUTPUT
%   figs : cell array di handle delle figure create

xb = iter_trace.x_before;
assert(numel(xb) == 2, 'plot_cg_storyboard richiede n=2.');
cgt = iter_trace.cg_trace;
nJ = numel(cgt);

if ~isempty(figdir) && ~exist(figdir,'dir'), mkdir(figdir); end
step_dir = fullfile(figdir, sprintf('iter%02d', k_idx));
if ~isempty(figdir) && ~exist(step_dir,'dir'), mkdir(step_dir); end

figs = {};
panel_no = 0;

% --- griglia di sfondo (contour) riutilizzata per tutti i pannelli ---
g0 = gradf(xb);
Zpts_all = [zeros(2,1), reshape([cgt.z], 2, nJ)];
allpts = xb + Zpts_all;
span_x = max(allpts(1,:)) - min(allpts(1,:));
span_y = max(allpts(2,:)) - min(allpts(2,:));
span = max([span_x, span_y, norm(g0)*0.5, 0.5]);
xr = linspace(xb(1)-1.2*span, xb(1)+1.2*span, 220);
yr = linspace(xb(2)-1.2*span, xb(2)+1.2*span, 220);
[X,Y] = meshgrid(xr,yr);
Zsurf = zeros(size(X));
for i=1:numel(xr)
    for jj=1:numel(yr)
        Zsurf(jj,i) = f([X(jj,i); Y(jj,i)]);
    end
end

    function draw_background()
        hold on; grid on; box on;
        zpos = Zsurf(Zsurf>0);
        if ~isempty(zpos)
            contour(X,Y,Zsurf, logspace(log10(max(min(zpos),1e-8)), log10(max(zpos)), 25), 'LineColor',[0.82 0.82 0.82]);
        else
            contour(X,Y,Zsurf,25,'LineColor',[0.82 0.82 0.82]);
        end
        plot(xb(1), xb(2), 'k^', 'MarkerFaceColor','g', 'MarkerSize', 10, 'DisplayName','x_k');
        xlabel('x_1'); ylabel('x_2'); axis equal;
    end

    function save_panel(fig, name)
        if ~isempty(figdir)
            fname = matlab.lang.makeValidName(name);
            exportgraphics(fig, fullfile(step_dir, [fname '.png']), 'Resolution', 200);
        end
        figs{end+1} = fig; %#ok<AGROW>
    end

%% --- Pannello 0: direzione iniziale = -grad f(x_k) ---
panel_no = panel_no + 1;
fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.55 0.55]);
draw_background();
d0 = -g0/norm(g0)*span*0.6;
quiver(xb(1), xb(2), d0(1), d0(2), 0, 'Color',[0.85 0.2 0.2], 'LineWidth',2.2, 'MaxHeadSize',0.6, 'DisplayName','d_0 = -\nabla f(x_k)');
title({sprintf('Iter %d - Step 0: si fissa la direzione iniziale', k_idx), ...
       'Il CG parte SEMPRE dalla direzione di steepest descent'}, 'FontSize',10,'Interpreter','tex');
legend('Location','best');
save_panel(fig, sprintf('%s_%s_iter%02d_panel00_steepest', method_name, problem_name, k_idx));

%% --- Pannelli 1..nJ: ogni step del CG ---
cmapJ = winter(max(nJ,1));
for j = 1:nJ
    panel_no = panel_no + 1;
    fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.6 0.6]);
    draw_background();

    z_prev = Zpts_all(:, j);
    p_prev = xb + z_prev;
    d_j = cgt(j).dk;
    curv = cgt(j).curv;
    thresh = cgt(j).curv_thresh;
    decision = cgt(j).decision;

    % direzione testata d_j, disegnata da p_prev
    d_scaled = d_j/norm(d_j)*span*0.5;
    quiver(p_prev(1), p_prev(2), d_scaled(1), d_scaled(2), 0, ...
           'Color',[0.3 0.3 0.9], 'LineWidth',1.8, 'MaxHeadSize',0.6, ...
           'LineStyle','--', 'DisplayName', sprintf('d_%d testata',j));

    if strcmp(decision, 'stopped_curvature')
        % --- TEST DI CURVATURA FALLITO ---
        plot(p_prev(1), p_prev(2), 'ks', 'MarkerFaceColor',[1 0.6 0], 'MarkerSize', 11, ...
             'DisplayName', 'ultima z buona (mantenuta)');
        title({sprintf('Iter %d - Step %d: TEST DI CURVATURA', k_idx, j), ...
               sprintf('d_j^T H d_j = %.3e \\leq soglia %.3e  \\rightarrow  curvatura non positiva', curv, thresh), ...
               'STOP: si mantiene l''ultima direzione buona (o -grad se j=1)'}, ...
               'FontSize',9.5,'Interpreter','tex','Color',[0.7 0.2 0]);
    else
        % --- passo accettato: calcolo nuova z ---
        z_curr = Zpts_all(:, j+1);
        p_curr = xb + z_curr;
        quiver(p_prev(1), p_prev(2), p_curr(1)-p_prev(1), p_curr(2)-p_prev(2), 0, ...
               'Color', cmapJ(j,:), 'LineWidth', 2.2, 'MaxHeadSize', 0.6, ...
               'DisplayName', sprintf('z_%d (nuova iterata)', j));
        plot(p_curr(1), p_curr(2), 'o', 'Color', cmapJ(j,:), 'MarkerFaceColor', cmapJ(j,:), 'MarkerSize', 8);

        if strcmp(decision, 'stopped_tol')
            crit_txt = sprintf('||r_%d|| \\leq \\eta  \\rightarrow  criterio soddisfatto, STOP (p_k = z_%d)', j, j);
            col = [0.1 0.55 0.1];
        else
            crit_txt = sprintf('||r_%d|| > \\eta  \\rightarrow  si continua con un altro step CG', j);
            col = [0.2 0.2 0.6];
        end

        title({sprintf('Iter %d - Step %d: CURVATURA POSITIVA (%.3e > %.3e)', k_idx, j, curv, thresh), ...
               'si accetta il passo e si aggiorna z', crit_txt}, ...
               'FontSize',9.5,'Interpreter','tex','Color',col);
    end

    legend('Location','best');
    save_panel(fig, sprintf('%s_%s_iter%02d_panel%02d_cgstep%d', method_name, problem_name, k_idx, panel_no, j));
end

%% --- Pannello finale: direzione accettata p_k ---
panel_no = panel_no + 1;
fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.55 0.55]);
draw_background();
pk = iter_trace.pk;
pk_scaled = pk/norm(pk)*span*0.7;
quiver(xb(1), xb(2), pk_scaled(1), pk_scaled(2), 0, 'Color',[0.6 0 0.6], 'LineWidth', 2.5, ...
       'MaxHeadSize', 0.5, 'DisplayName', 'p_k finale (direzione di ricerca)');
title({sprintf('Iter %d - Direzione finale accettata (uscita CG: %s)', k_idx, iter_trace.cg_exit), ...
       sprintf('||p_k|| = %.4e', iter_trace.pk_norm)}, 'FontSize',10,'Interpreter','none');
legend('Location','best');
save_panel(fig, sprintf('%s_%s_iter%02d_panelFinal_pk', method_name, problem_name, k_idx));

%% --- Pannello Armijo: curva REALE phi(alpha) vs soglia ---
panel_no = panel_no + 1;
fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.55 0.45]);
hold on; grid on; box on;

alpha_grid = linspace(0, 1.15, 200);
phi = zeros(size(alpha_grid));
for i = 1:numel(alpha_grid)
    phi(i) = f(xb + alpha_grid(i)*pk);
end
g_pk = gradf(xb)' * pk; % coefficiente angolare esatto della retta di Armijo (derivata direzionale)
c1_used = (iter_trace.bt_trace(1).armijo_rhs - iter_trace.fk) / (iter_trace.bt_trace(1).alpha * g_pk);
armijo_line = iter_trace.fk + c1_used * g_pk * alpha_grid;

plot(alpha_grid, phi, 'b-', 'LineWidth', 2, 'DisplayName', '\phi(\alpha) = f(x_k+\alpha p_k)  [valore REALE]');
plot(alpha_grid, armijo_line, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('soglia Armijo (c_1=%.1e)', c1_used));

bt = iter_trace.bt_trace;
for t = 1:numel(bt)
    if bt(t).accepted
        col = [0.1 0.7 0.1]; mk = 'o';
    else
        col = [0.8 0.1 0.1]; mk = 'x';
    end
    plot(bt(t).alpha, bt(t).f_trial, mk, 'Color', col, 'MarkerFaceColor', col, ...
         'MarkerSize', 9, 'LineWidth', 2, 'HandleVisibility','off');
    text(bt(t).alpha, bt(t).f_trial, sprintf('  t=%d, \\alpha=%.3g', t, bt(t).alpha), ...
         'FontSize', 7.5, 'Color', col*0.8);
end
plot(NaN,NaN,'o','Color',[0.1 0.7 0.1],'MarkerFaceColor',[0.1 0.7 0.1],'DisplayName','tentativo accettato');
plot(NaN,NaN,'x','Color',[0.8 0.1 0.1],'LineWidth',2,'DisplayName','tentativo rifiutato');

xlabel('\alpha'); ylabel('valore di f lungo la direzione p_k');
title(sprintf('Iter %d - Condizione di Armijo: \\phi(\\alpha) confrontata con la soglia', k_idx), ...
      'FontSize',10,'Interpreter','tex');
legend('Location','best');

save_panel(fig, sprintf('%s_%s_iter%02d_panelArmijo', method_name, problem_name, k_idx));

end
