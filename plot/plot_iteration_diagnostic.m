function fig = plot_iteration_diagnostic(f, gradf, iter_trace, k_idx, method_name, problem_name, figdir)
% PLOT_ITERATION_DIAGNOSTIC - per UNA iterazione esterna k, mostra:
%
%   PANNELLO SINISTRO (evoluzione del CG):
%     contour locale attorno a x_k, con una freccia per ogni iterata
%     parziale z_j costruita dal CG (z_1, z_2, ..., fino a p_k). La
%     prima freccia (j=1) e' sempre nella direzione -grad f(x_k)/||.||
%     scalata (il primo passo di CG e' sempre lungo lo steepest
%     descent), le successive mostrano come il CG "piega" la direzione
%     verso quella di Newton. L'ultima freccia e' evidenziata (p_k
%     finale) e annotata col motivo di uscita dal CG.
%
%   PANNELLO DESTRO (Armijo/backtracking):
%     asse x = alpha (scala log, decrescente: 1, rho, rho^2, ...),
%     asse y = valore di f. Punti = f(x_k + alpha*p_k) per ogni alpha
%     provato; linea = soglia di Armijo f(x_k) + c1*alpha*(grad'*p_k).
%     Punti verdi = accettati (f <= soglia), rossi = rifiutati.
%
% INPUT
%   f, gradf     : function handle (n=2) usati per disegnare/verificare
%   iter_trace   : trace(k_idx) prodotto da
%                  truncated_newton_method_diagnostic.m (deve avere i
%                  campi x_before, pk, cg_trace, bt_trace, cg_exit, ecc.)
%   k_idx        : indice dell'iterazione esterna (solo per titolo/nome file)
%   method_name, problem_name, figdir : per titolo/nome file

xb = iter_trace.x_before;
assert(numel(xb) == 2, 'plot_iteration_diagnostic richiede n=2.');

fig = figure('Color','w','Units','normalized','Position',[0.05 0.1 0.85 0.5]);

%% --- PANNELLO SINISTRO: evoluzione CG ---
subplot(1,2,1); hold on; grid on; box on;

cgt = iter_trace.cg_trace;
nJ = numel(cgt);

% costruisco tutte le iterate incluso z_0=0 (=xb stesso)
Zpts = [zeros(2,1), reshape([cgt.z], 2, nJ)]; % 2 x (nJ+1), colonna in coordinate RELATIVE a xb
% range assi basato sull'inviluppo di tutte le iterate + xb
allpts = xb + Zpts;
xr = linspace(min(allpts(1,:))-0.3, max(allpts(1,:))+0.3, 200);
yr = linspace(min(allpts(2,:))-0.3, max(allpts(2,:))+0.3, 200);
if range(xr) < 1e-6, xr = xb(1) + linspace(-1,1,200); end
if range(yr) < 1e-6, yr = xb(2) + linspace(-1,1,200); end
[X,Y] = meshgrid(xr,yr);
Z = zeros(size(X));
for i=1:numel(xr)
    for jj=1:numel(yr)
        Z(jj,i) = f([X(jj,i); Y(jj,i)]);
    end
end
zpos = Z(Z>0);
if ~isempty(zpos)
    contour(X,Y,Z, logspace(log10(max(min(zpos),1e-8)), log10(max(zpos)), 25), 'LineColor',[0.8 0.8 0.8]);
else
    contour(X,Y,Z,25,'LineColor',[0.8 0.8 0.8]);
end

plot(xb(1), xb(2), 'k^', 'MarkerFaceColor','g', 'MarkerSize', 10, 'DisplayName','x_k');

cmapJ = winter(max(nJ,1));
for j = 1:nJ
    z_prev = Zpts(:, j);   % iterata precedente (relativa)
    z_curr = Zpts(:, j+1); % iterata corrente (relativa)
    p_prev = xb + z_prev;
    p_curr = xb + z_curr;

    is_last = (j == nJ);
    lw = 1.5 + 1.5*is_last;
    quiver(p_prev(1), p_prev(2), p_curr(1)-p_prev(1), p_curr(2)-p_prev(2), 0, ...
           'Color', cmapJ(j,:), 'LineWidth', lw, 'MaxHeadSize', 0.6, ...
           'HandleVisibility','off');
    plot(p_curr(1), p_curr(2), 'o', 'Color', cmapJ(j,:), 'MarkerFaceColor', cmapJ(j,:), ...
         'MarkerSize', 4+2*is_last, 'HandleVisibility','off');

    if j==1
        txt = sprintf('j=1 (steepest descent)\ncurv=%.2e', cgt(j).curv);
    else
        txt = sprintf('j=%d, curv=%.2e\n%s', j, cgt(j).curv, cgt(j).decision);
    end
    text(p_curr(1), p_curr(2), ['  ' txt], 'FontSize', 7, 'Color', cmapJ(j,:)*0.7);
end

xlabel('x_1'); ylabel('x_2');
title(sprintf('CG interno, iter %d: z_j costruita dal CG (uscita: %s)', k_idx, iter_trace.cg_exit), ...
      'FontSize', 9, 'Interpreter','none');
legend('Location','best');

%% --- PANNELLO DESTRO: Armijo / backtracking ---
subplot(1,2,2); hold on; grid on; box on;

bt = iter_trace.bt_trace;
alphas = [bt.alpha];
ftrials = [bt.f_trial];
armijo_rhs = [bt.armijo_rhs];
accepted = [bt.accepted];

[alphas_sorted, order] = sort(alphas, 'descend');
ftrials_sorted = ftrials(order);
armijo_sorted  = armijo_rhs(order);
accepted_sorted = accepted(order);

% soglia di Armijo come funzione continua di alpha (retta in questo caso,
% dato che f(xk)+c1*alpha*g'p e' lineare in alpha)
alpha_grid = linspace(0, max(alphas)*1.05, 100);
% ricostruzione retta: passa per (alpha_i, armijo_rhs_i) - lineare in alpha
if numel(alphas) >= 1
    % f(xk) + c1*(g'p)*alpha -> ricavo c1*(g'p) dal primo punto
    slope = (armijo_rhs(1) - iter_trace.fk) / alphas(1);
    armijo_line = iter_trace.fk + slope * alpha_grid;
    plot(alpha_grid, armijo_line, 'k--', 'LineWidth', 1.2, 'DisplayName','soglia Armijo');
end

for t = 1:numel(alphas_sorted)
    if accepted_sorted(t)
        col = [0.1 0.7 0.1]; mk = 'o'; lbl = 'accettato';
    else
        col = [0.8 0.1 0.1]; mk = 'x'; lbl = 'rifiutato';
    end
    plot(alphas_sorted(t), ftrials_sorted(t), mk, 'Color', col, ...
         'MarkerFaceColor', col, 'MarkerSize', 8, 'LineWidth', 1.5, 'HandleVisibility','off');
    text(alphas_sorted(t), ftrials_sorted(t), sprintf('  \\alpha=%.4f', alphas_sorted(t)), ...
         'FontSize', 7, 'Color', col*0.8);
end
% marker fittizi per la legenda
plot(NaN,NaN,'o','Color',[0.1 0.7 0.1],'MarkerFaceColor',[0.1 0.7 0.1],'DisplayName','accettato (f \leq soglia)');
plot(NaN,NaN,'x','Color',[0.8 0.1 0.1],'LineWidth',1.5,'DisplayName','rifiutato (f > soglia)');

set(gca, 'XScale', 'log');
xlabel('\alpha (scala log)'); ylabel('valore di f');
title(sprintf('Backtracking/Armijo, iter %d: %d tentativi', k_idx, numel(bt)), ...
      'FontSize', 9, 'Interpreter','none');
legend('Location','best');

sgtitle(sprintf('%s - %s: dettaglio iterazione %d', method_name, problem_name, k_idx), ...
        'Interpreter','none');

if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('iterdiag_%s_%s_iter%02d', method_name, problem_name, k_idx));
    exportgraphics(fig, fullfile(figdir, [fname '.png']), 'Resolution', 200);
end

end
