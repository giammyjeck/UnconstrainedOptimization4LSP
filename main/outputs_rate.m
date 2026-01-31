function figRate = outputs_rate(dims, labels, res_method, fk_vec, method, figdir, pname)
% OUTPUTS_RATE
% Plotta i tassi di convergenza stimati per ogni run
% Accanto all'ultimo punto: valore f(x_final) preso da fk_vec
%
% INPUTS:
% - dims: dimensione del problema (non usata nel grafico)
% - labels: nomi dei run
% - res_method: struct contenente xseq e rate stimati per ogni label
% - fk_vec: valori finali della funzione obiettivo per ogni run
% - method: nome del metodo (per titolo figura)
% - figdir: cartella dove salvare la figura
% - pname: nome del file da salvare

figRate = [];

if isempty(labels)
    warning('Labels vuote, niente da plottare');
    return;
end

%% === Figura ===
figRate = figure('Name', [method ' - Rate di convergenza'], ...
                 'Color', 'w', 'Units', 'normalized', ...
                 'Position', [0.1 0.1 0.75 0.55]);
hold on; grid on;
colors = lines(numel(labels));

%% === Plot per ogni run ===
for s = 1:numel(labels)
    label = labels{s};
    rate_exp = res_method.(label).rate;  % vettore dei rate stimati
    K = numel(rate_exp);
    
    % curva del rate stimato
    plot(1:K, rate_exp, '-o', 'Color', colors(s,:), ...
         'LineWidth', 1.4, 'MarkerSize', 4);
    

end

%% === Labels e titolo ===
xlabel('Iterazione k');
ylabel('Rate esponenziale stimato');
title(method);

%% === Salvataggio ===
if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    name = pname + ".png";
    exportgraphics(figRate, fullfile(figdir, name));
end

end
