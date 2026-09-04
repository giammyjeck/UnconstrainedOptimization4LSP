function H = trig_hess_fd_case1(gradf, x, k, mode)
% Approssima l'Hessiana (diagonale) di Trig16 per differenze finite
% CENTRATE, usando il gradiente (esatto o FD) fornito.
%
% Perche' funziona: la Hessiana di Trig16 e' diagonale, e ogni componente
% del gradiente g_i dipende SOLO da x_i (non dai vicini). Per questo
% possiamo perturbare tutte le componenti di x contemporaneamente con il
% vettore di step h, e leggere H_ii come differenza centrata di g_i,
% senza bisogno di perturbare una variabile alla volta (che sarebbe O(n)
% valutazioni, troppo lento per n grande).
%
% NOTA SULLO STEP OTTIMALE (differenza centrata, non piu' in avanti):
% l'errore di troncamento e' ora O(h^2) (non O(h) come nella versione in
% avanti), mentre l'errore di cancellazione numerica resta O(eps_mach/h)
% (la centrata non peggiora la cancellazione rispetto all'avanti: si
% sottraggono comunque due valori vicini, non tre come nella derivata
% seconda centrata di trig_fd_case2.m). Il passo ottimale che minimizza
% la somma dei due e' h_ott ~ eps_mach^(1/3) ~ 6e-6, cioe' k~5, quindi
% leggermente diverso sia da k=8 (ottimale per l'avanti) sia da k=4
% (ottimale per la differenza SECONDA centrata di case2, che ha invece
% cancellazione O(eps_mach/h^2)). Ci si aspetta un errore minimo un
% ordine di grandezza migliore rispetto alla versione in avanti, al
% prezzo del doppio delle valutazioni di gradiente (due invece di una).
 
    x = x(:);
    n = length(x);
 
    h = steps_fd(x, k, mode);
 
    gp = gradf(x + h);   % gradiente perturbato in avanti
    gm = gradf(x - h);   % gradiente perturbato indietro
 
    d = (gp - gm) ./ (2*h);   % differenza centrata componente per componente
    % d_i approssima dg_i/dx_i = H_ii
 
    H = spdiags(d, 0, n, n);  % Hessiana diagonale sparsa
end