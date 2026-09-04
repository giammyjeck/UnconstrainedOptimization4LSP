function H = trig_hess_fd_case1_avanti(gradf, x, k, mode)
% Approssima l'Hessiana (diagonale) di Trig16 per
% differenze finite in AVANTI, usando il gradiente (esatto o FD) fornito.
%
% Perche' funziona: la Hessiana di Trig16 e' diagonale, e ogni componente
% del gradiente g_i dipende SOLO da x_i (non dai vicini). Per questo
% possiamo perturbare tutte le componenti di x contemporaneamente con il
% vettore di step h, e leggere H_ii come differenza in avanti di g_i,
% senza bisogno di perturbare una variabile alla volta (che sarebbe O(n)
% valutazioni, troppo lento per n grande).
%
% NOTA SULLO STEP OTTIMALE: essendo una differenza in AVANTI (non
% centrata), l'errore di troncamento e' O(h), mentre l'errore di
% cancellazione numerica e' O(eps_mach/h). Il passo ottimale che minimizza
% la somma dei due e' h_ott ~ sqrt(eps_mach) ~ 1.5e-8, cioe' k=8 -- non
% k=4 come nel caso Case 2 (vedi trig_fd_case2.m), che usa differenze
% CENTRATE con errore di troncamento O(h^2) e quindi ottimale a k=4.
% Verificato numericamente: errore ~5e-4 a k=4, ~7e-8 a k=8 (minimo),
% torna a ~5e-4 a k=12 per cancellazione.

    x = x(:);
    n = length(x);

    h = steps_fd(x, k, mode);

    g0 = gradf(x);   %gradiente nel punto corrente 
    gp = gradf(x + h);  %gradiente perturbato (tutte le componenti insieme) 

    d = (gp - g0) ./ h; %differenza in avanti componente per componente 
% d_i approssima dg_i/dx_i = H_ii

    H = spdiags(d, 0, n, n);  % Hessiana diagonale sparsa
end