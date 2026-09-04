function fig = plot_gradnorm_vs_reference(xseq_list, gradf, method_name, problem_name, figdir, tail_len)
% PLOT_GRADNORM_VS_REFERENCE - traccia ||grad f(x_k)|| vs k (semilogy)
% per ciascun punto di partenza, sovrapponendo DUE CURVE DI RIFERIMENTO
% locali ancorate alla coda della sequenza:
%   - lineare:   e_k   = e0 * rho^k              (tratteggiata)
%   - quadratica: e_{k+1} = C * e_k^2            (punteggiata)
%
% A differenza degli stimatori puntuali (estimate_rate,
% estimate_rate_polyfit), qui non si calcola MAI un rapporto tra numeri
% vicini: si stima rho/C mediando su una finestra (media geometrica,
% equivalente a un fit robusto) e si genera una curva continua da
% confrontare VISIVAMENTE con l'andamento vero. Il giudizio sul rate
% diventa "la curva vera segue la tratteggiata o la punteggiata?"
% invece di leggere un numero rumoroso.
%
% Le curve di riferimento sono ANCORATE LOCALMENTE alla coda (ultimi
% tail_len punti) di ciascuna sequenza, non estrapolate su tutto il
% range: lontano dalla convergenza (fase transitoria, backtracking,
% correzione Hessiana) il concetto di "rate" non e' comunque valido, ed
% e' piu' onesto non disegnare li' nessun riferimento.
%
% INPUT
%   xseq_list  : cell array, uno per starting point, ciascuno n x K
%   gradf      : function handle per il gradiente (es. grad_exact)
%   method_name, problem_name, figdir : per titolo/nome file
%   tail_len   : (opzionale, default 5) numero di punti finali usati per
%                ancorare le curve di riferimento
%
% USO TIPICO:
%   plot_gradnorm_vs_reference(xseq_list, grad_exact, 'ModifiedNewton', 'Trig16', 'graphs_trig16');
 
if nargin < 6 || isempty(tail_len), tail_len = 5; end
 
fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.65 0.5]);
hold on; grid on; box on;
colors = lines(numel(xseq_list));
any_plotted = false;
 
for i = 1:numel(xseq_list)
    xseq = xseq_list{i};
    if isempty(xseq), continue; end
    K = size(xseq, 2);
 
    gnorm = zeros(1, K);
    for k = 1:K
        gnorm(k) = norm(gradf(xseq(:,k)));
    end
 
    kvec = 1:K;
    semilogy(kvec, gnorm, '-o', 'Color', colors(i,:), ...
             'LineWidth', 1.3, 'MarkerSize', 3, ...
             'DisplayName', sprintf('start %d', i));
    any_plotted = true;
 
    % --- curve di riferimento, ancorate alla coda ---
    if K < tail_len + 1, continue; end   % coda troppo corta per stimare rho/C in modo sensato
 
    tail_idx = (K - tail_len + 1):K;
    e_tail = gnorm(tail_idx);
 
    if any(e_tail <= 0), continue; end   % non dovrebbe succedere con norme di gradiente, ma per sicurezza
 
    k0 = tail_idx(1);   e0 = e_tail(1);
    kE = tail_idx(end); eE = e_tail(end);
    n  = kE - k0;
    if n < 1, continue; end
 
    k_ref = k0:kE;
    j_ref = k_ref - k0;   % 0, 1, ..., n
 
    % --- riferimento LINEARE: e_k = e0 * rho^(k-k0), ANCORATO a
    % entrambi gli estremi (rho scelto per passare esattamente per
    % (k0,e0) e (kE,eE)). Per costruzione e' compreso tra e0 ed eE,
    % non puo' "esplodere" fuori da quell'intervallo.
    rho = (eE/e0)^(1/n);
    e_lin = e0 * rho .^ j_ref;
 
    % --- riferimento QUADRATICO: e_{j+1} = C*e_j^2, telescopando si
    % ottiene e_j = C^(2^j-1) * e0^(2^j). Si sceglie C in modo che il
    % modello passi ESATTAMENTE per (kE,eE), non si estrapola in avanti
    % da un C stimato punto-per-punto (che e' la fonte dell'instabilita'
    % vista prima): anche questa curva e' quindi vincolata agli stessi
    % due punti reali agli estremi, e la sua forma INTERMEDIA (piu' o
    % meno ripida della lineare) e' l'unica cosa che dice se il
    % decadimento assomiglia di piu' a quadratico o lineare.
    Cq = (eE / e0^(2^n))^(1/(2^n - 1));
    e_quad = (Cq.^(2.^j_ref - 1)) .* (e0.^(2.^j_ref));
 
    semilogy(k_ref, e_lin,  '--', 'Color', colors(i,:), 'LineWidth', 1.0, 'HandleVisibility','off');
    semilogy(k_ref, e_quad, ':',  'Color', colors(i,:), 'LineWidth', 1.5, 'HandleVisibility','off');
end
 
if ~any_plotted
    warning('plot_gradnorm_vs_reference: nessuna sequenza valida per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    close(fig); fig = []; return;
end
 
set(gca, 'YScale', 'log');
xlabel('Iterazione k');
ylabel('||grad f(x_k)||');
title(sprintf('%s - %s: andamento gradiente vs riferimenti lineare (- -) e quadratico (\\cdot\\cdot\\cdot)', ...
      method_name, problem_name), 'Interpreter','tex');
legend('Location','best');
 
if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('gradnorm_ref_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end
 
end
 