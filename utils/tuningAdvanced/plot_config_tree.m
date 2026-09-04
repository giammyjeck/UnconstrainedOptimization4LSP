function plot_config_tree(registry, fig_title, highlight_keys)
% PLOT_CONFIG_TREE Disegna l'albero delle configurazioni esplorate
% durante il Refinement 1 (screening adattivo), mostrando le relazioni
% genitore -> figlio create dalle regole di escalation (bt, c1, kmax,
% max_cg) e colorando i nodi in base all'esito:
%   verde   = accettata (passata tutti gli starting point)
%   arancio = ha generato almeno una configurazione figlia (escalation)
%   rosso   = scartata (non converge e nessuna escalation applicabile,
%             oppure tutte le escalation possibili erano gia' state
%             testate)
%   grigio  = nodo radice della griglia iniziale (nessun genitore)
%
% registry       : containers.Map costruito con register_node/set_status
% fig_title      : titolo della figura
% highlight_keys : (opzionale) cell array di chiavi da evidenziare, es.
%                  le configurazioni finali selezionate dopo il
%                  Refinement 2 (marker a stella magenta)
 
    if nargin < 3
        highlight_keys = {};
    end
 
    all_keys = keys(registry);
    n = numel(all_keys);
    if n == 0
        warning('plot_config_tree: registro vuoto, nessun nodo da disegnare.');
        return;
    end
 
    src = {};
    tgt = {};
    for i = 1:n
        node = registry(all_keys{i});
        if ~isempty(node.parent_key)
            src{end+1} = node.parent_key; %#ok<AGROW>
            tgt{end+1} = all_keys{i};      %#ok<AGROW>
        end
    end
 
    if ~isempty(src)
        G = digraph(src, tgt);
    else
        G = digraph();
    end
    missing_nodes = setdiff(all_keys, G.Nodes.Name);
    if ~isempty(missing_nodes)
        G = addnode(G, missing_nodes);
    end
 
    nn = numnodes(G);
    colors = repmat([0.6 0.6 0.6], nn, 1); % default: grigio (nodo non nel registro / radice)
    labels = cell(nn, 1);
    for i = 1:nn
        name = G.Nodes.Name{i};
        labels{i} = name; % fallback se il nodo non e' nel registro
        if isKey(registry, name)
            node = registry(name);
 
            % Etichetta: completa per i nodi radice (griglia iniziale,
            % nessun genitore), solo i parametri CAMBIATI rispetto al
            % genitore per tutti gli altri, cosi' l'albero resta
            % leggibile anche con molti livelli di escalation.
            if isempty(node.parent_key) || ~isKey(registry, node.parent_key)
                labels{i} = node.short_label;
            else
                parent_node = registry(node.parent_key);
                labels{i} = make_diff_label(node.cfg, parent_node.cfg);
            end
 
            switch node.status
                case 'accepted'
                    colors(i,:) = [0.20 0.65 0.20]; % verde
                case 'escalated'
                    colors(i,:) = [0.90 0.60 0.10]; % arancio
                case 'discarded'
                    colors(i,:) = [0.80 0.20 0.20]; % rosso
                otherwise
                    colors(i,:) = [0.55 0.55 0.55]; % queued / sconosciuto
            end
        end
    end
 
    figure('Name', fig_title);
    h = plot(G, 'Layout', 'layered', 'NodeLabel', labels, ...
        'NodeColor', colors, 'MarkerSize', 7, 'ArrowSize', 9, ...
        'NodeFontSize', 7, 'EdgeColor', [0.5 0.5 0.5]);
    title(fig_title);
    axis off;
 
    if ~isempty(highlight_keys)
        valid_hl = intersect(highlight_keys, G.Nodes.Name);
        if ~isempty(valid_hl)
            highlight(h, valid_hl, 'Marker', 'p', 'MarkerSize', 16, ...
                'NodeColor', [0.85 0.10 0.85]);
        end
    end
 
    % Legenda manuale (il plot di un digraph non genera una legenda
    % automatica per i colori dei nodi).
    hold on;
    lg_x = xlim; lg_y = ylim;
    px = lg_x(1); py = lg_y(2);
    legend_items = {
        'accettata',        [0.20 0.65 0.20];
        'escalation',       [0.90 0.60 0.10];
        'scartata',         [0.80 0.20 0.20];
        'radice (griglia)', [0.55 0.55 0.55]
    };
    hleg = gobjects(size(legend_items,1),1);
    for i = 1:size(legend_items,1)
        hleg(i) = scatter(NaN, NaN, 60, legend_items{i,2}, 'filled');
    end
    if ~isempty(highlight_keys)
        hleg(end+1) = scatter(NaN, NaN, 100, [0.85 0.10 0.85], 'p', 'filled');
        legend_items(end+1,:) = {'selezionata (finale)', []};
    end
    legend(hleg, legend_items(:,1), 'Location', 'bestoutside');
    hold off;
end
 