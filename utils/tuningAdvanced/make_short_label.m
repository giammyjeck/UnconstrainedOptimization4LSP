function lbl = make_short_label(cfg)
% MAKE_SHORT_LABEL Costruisce un'etichetta compatta multi-riga per una
% configurazione, usata come node label nel plot ad albero. Include solo
% i campi effettivamente presenti in cfg, cosi' la stessa funzione va
% bene sia per le config del Modified Newton (con beta) sia per quelle
% del Truncated Newton (con max_cg).
 
    parts = {};
    if isfield(cfg, 'rho'),    parts{end+1} = sprintf('\\rho=%.2g', cfg.rho);      end
    if isfield(cfg, 'beta'),   parts{end+1} = sprintf('\\beta=%.0e', cfg.beta);    end
    if isfield(cfg, 'c1'),     parts{end+1} = sprintf('c1=%.0e', cfg.c1);          end
    if isfield(cfg, 'kmax'),   parts{end+1} = sprintf('kmax=%d', cfg.kmax);        end
    if isfield(cfg, 'bt'),     parts{end+1} = sprintf('bt=%d', cfg.bt);            end
    if isfield(cfg, 'max_cg'), parts{end+1} = sprintf('cg=%d', cfg.max_cg);        end
 
    lbl = strjoin(parts, newline);
end
 