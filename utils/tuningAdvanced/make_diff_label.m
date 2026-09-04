function lbl = make_diff_label(cfg, parent_cfg)
% MAKE_DIFF_LABEL Costruisce un'etichetta compatta che mostra SOLO i
% parametri cambiati rispetto alla configurazione genitrice (parent_cfg).
% Usata per i nodi non radice dell'albero delle configurazioni, cosi'
% l'etichetta di ogni nodo mostra a colpo d'occhio "cosa e' stato
% modificato dall'escalation", invece di ripetere tutti i parametri
% (che nella maggior parte dei casi restano identici al genitore).
%
% cfg        : struct della configurazione corrente
% parent_cfg : struct della configurazione genitrice
 
    field_order = {'rho', 'beta', 'c1', 'kmax', 'bt', 'max_cg'};
    parts = {};
 
    for i = 1:numel(field_order)
        fn = field_order{i};
        if ~isfield(cfg, fn)
            continue;
        end
        changed = ~isfield(parent_cfg, fn) || cfg.(fn) ~= parent_cfg.(fn);
        if changed
            parts{end+1} = format_param(fn, cfg.(fn)); %#ok<AGROW>
        end
    end
 
    if isempty(parts)
        % Non dovrebbe succedere (chiavi diverse -> almeno un parametro
        % diverso), ma per sicurezza mostriamo comunque qualcosa.
        lbl = '(nessun cambio rilevato)';
    else
        lbl = strjoin(parts, newline);
    end
end
 
 
function s = format_param(fn, val)
    switch fn
        case 'rho',    s = sprintf('\\rho=%.2g', val);
        case 'beta',   s = sprintf('\\beta=%.0e', val);
        case 'c1',     s = sprintf('c1=%.0e', val);
        case 'kmax',   s = sprintf('kmax=%d', val);
        case 'bt',     s = sprintf('bt=%d', val);
        case 'max_cg', s = sprintf('cg=%d', val);
        otherwise,     s = sprintf('%s=%g', fn, val);
    end
end
 