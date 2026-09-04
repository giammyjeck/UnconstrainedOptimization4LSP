function [rate_seq, k_centers] = estimate_rate_polyfit(err, window, min_err)
% ESTIMATE_RATE_POLYFIT - stima il rate di convergenza p tale che
%   e_{k+1} ~ C * e_k^p
% usando una regressione lineare (polyfit) su una finestra scorrevole di
% punti consecutivi, in scala log-log:
%   log(e_{k+1}) = log(C) + p * log(e_k)
% La pendenza della retta e' la stima di p per quella finestra.
%
% Molto piu robusta dello stimatore a 3 punti (log(r_k)/log(r_{k-1})),
% perche' mediando su piu' osservazioni riduce l'amplificazione del
% rumore tipica della divisione tra due logaritmi quasi uguali.
%
% NON va applicata su tutta la sequenza in un colpo solo se la sequenza
% attraversa piu' regimi di convergenza (es. transiente + quadratico +
% lineare finale): mescolerebbe le pendenze in un unico valore medio
% privo di significato per ciascun regime. Usa una finestra scorrevole
% per vedere come il rate CAMBIA lungo la sequenza.
%
% INPUT
%   err     : [1 x K] sequenza di errori (es. ||x_k - x*|| oppure
%             ||grad f(x_k)||), STRETTAMENTE POSITIVA dove usata
%   window  : numero di punti per ogni finestra di regressione
%             (consigliato 4-6; troppo piccola torna rumorosa come lo
%             stimatore a 3 punti, troppo grande mescola i regimi)
%   min_err : soglia sotto la quale un valore di errore e' considerato
%             rumore numerico e va escluso dal fit (default 1e-12)
%
% OUTPUT
%   rate_seq  : [1 x M] rate stimato per ogni finestra
%   k_centers : [1 x M] indice di iterazione al centro di ciascuna
%               finestra, utile per plottare rate_seq vs iterazione

if nargin < 3, min_err = 1e-12; end

err = err(:)';
K = numel(err);

% maschera dei punti "validi" (sopra la soglia di rumore)
valid = err > min_err;

rate_seq  = [];
k_centers = [];

if window < 3
    error('window deve essere almeno 3 per una regressione sensata');
end

for k0 = 1:(K - window + 1)
    idx = k0 : k0+window-1;
    if ~all(valid(idx)), continue; end   % salta finestre che toccano rumore

    x_win = log(err(idx(1:end-1)));   % log(e_k)
    y_win = log(err(idx(2:end)));     % log(e_{k+1})

    if numel(x_win) < 2, continue; end

    pfit = polyfit(x_win, y_win, 1);  % pfit(1) = pendenza = rate stimato
    rate_seq(end+1)  = pfit(1); %#ok<AGROW>
    k_centers(end+1) = idx(1) + floor(window/2); %#ok<AGROW>
end

end
