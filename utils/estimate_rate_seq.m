function [rate_seq, k_idx, rate_raw, flag_stagn, flag_noise] = estimate_rate_seq(err, window, min_err, min_logratio)
% ESTIMATE_RATE_SEQ - stima l'ordine di convergenza q con la formula
% puntuale a 3 punti (eq. 33/34 delle dispense):
%
%       p_k = log( e_k / e_{k-1} ) / log( e_{k-1} / e_{k-2} )
%
% aggregata con mediana mobile robusta (non un fit: non puo' mai essere
% mal condizionata).
%
% Una tripletta viene ESCLUSA dalla stima (rate_raw(k-2) = NaN) in due
% casi DISTINTI, tenuti separati negli output cosi' da poterli
% distinguere nel grafico:
%
%   - flag_noise(k-2)  = true : uno o piu' degli err coinvolti e'
%     sotto min_err -> rumore numerico/precisione macchina.
%
%   - flag_stagn(k-2)  = true : |log(e_k/e_{k-1})| oppure
%     |log(e_{k-1}/e_{k-2})| e' sotto min_logratio, cioe' due errori
%     consecutivi sono quasi identici in RAPPORTO (non in valore
%     assoluto: possono benissimo essere ben sopra min_err). Qui la
%     formula (33) diventa 0/qualcosa o qualcosa/0: matematicamente
%     definita ma priva di significato come "ordine di convergenza"
%     per quella tripletta. Questo e' il caso da ispezionare per
%     capire se e' rumore dello stimatore oppure un vero segnale
%     dell'algoritmo (backtracking pesante, correzione dell'Hessiana
%     che cresce, ecc.) - per questo va segnalato, non solo scartato.
%
% INPUT
%   err          : [1 x K] sequenza di errori, positiva dove usata
%   window       : (opzionale, default 5) ampiezza mediana mobile su p_k
%   min_err      : (opzionale, default 1e-10) soglia di rumore numerico
%   min_logratio : (opzionale, default 1e-3) soglia sotto la quale un
%                  log-rapporto e' considerato "quasi zero"
%
% OUTPUT
%   rate_seq   : [1 x K-2] stima aggregata (mediana mobile) di q
%   k_idx      : [1 x K-2] indici di iterazione corrispondenti
%   rate_raw   : [1 x K-2] stima puntuale grezza (prima dell'aggregazione)
%   flag_stagn : [1 x K-2] logical, true dove scartato per log-ratio~0
%   flag_noise : [1 x K-2] logical, true dove scartato per rumore numerico
%
% USO TIPICO:
%   [rate_seq, k_idx, ~, flag_stagn] = estimate_rate_seq(err, 5);
%   plot(k_idx, rate_seq, '-o'); hold on;
%   plot(k_idx(flag_stagn), zeros(1,nnz(flag_stagn)), 'rx');
 
if nargin < 2 || isempty(window),       window       = 5;    end
if nargin < 3 || isempty(min_err),      min_err      = 1e-10; end
if nargin < 4 || isempty(min_logratio), min_logratio = 1e-3; end
 
err = err(:)';
K   = numel(err);
 
if K < 3
    rate_seq = []; k_idx = []; rate_raw = []; flag_stagn = []; flag_noise = [];
    return;
end
 
rate_raw   = nan(1, K-2);
flag_stagn = false(1, K-2);
flag_noise = false(1, K-2);
k_idx      = 3:K;
 
for k = 3:K
    idx = k-2;
    e1 = err(k-2); e2 = err(k-1); e3 = err(k);
 
    if e1 <= min_err || e2 <= min_err || e3 <= min_err
        flag_noise(idx) = true;
        continue;
    end
 
    log_num = log(e3/e2);
    log_den = log(e2/e1);
 
    if abs(log_num) < min_logratio || abs(log_den) < min_logratio
        flag_stagn(idx) = true;
        continue;
    end
 
    rate_raw(idx) = log_num / log_den;
end
 
if window <= 1
    rate_seq = rate_raw;
else
    rate_seq = movmedian(rate_raw, window, 'omitnan');
end
 
end
 