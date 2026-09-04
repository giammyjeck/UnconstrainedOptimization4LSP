function [g, H] = trig_fd_case2(x, k, mode)
% IDEA CHIAVE: riscriviamo F(x) come somma di termini phi_i(x_i), ciascuno
% dipendente da UNA SOLA variabile:
%   F(x) = sum_i phi_i(x_i),   phi_i(x_i) = i*(1-cos(x_i)) + c_i*sin(x_i)
% con c_i = 2 per i<n e c_i = -(n-1) per i=n (si ottiene raggruppando per
% indice i tutti i termini della formula originale di F che coinvolgono
% x_i). Verificato: sum(phi_i) = F(x) esattamente.
%
% Questo E' DIVERSO (e corretto, a differenza) dall'uso dei residui r_i
% del problema (vedi problem_trig16.m), che dipendono anche da x_{i-1} e
% x_{i+1}: usare quelli con lo stesso schema vettorizzato darebbe un
% gradiente SBAGLIATO, perche' servirebbe sommare i contributi di TRE
% residui per ogni g_i, non uno solo. Qui invece, siccome phi_i dipende
% solo da x_i, dF/dx_i = phi_i'(x_i) esattamente, quindi possiamo
% perturbare tutte le x insieme e leggere il gradiente componente per
% componente in un solo colpo (O(1) valutazioni vettoriali, non O(n)).
%
% ATTENZIONE - INSTABILITA' NUMERICA ATTESA PER k GRANDE (h piccolo):
% la formula dell'Hessiana e' una differenza seconda CENTRATA, che divide
% per h^2. Per k=8 o k=12 (h=1e-8 o 1e-12), h^2 e' dell'ordine o sotto la
% precisione di macchina (eps_mach ~2.22e-16), quindi il risultato e'
% dominato da rumore numerico, non e' un bug. Il passo ottimale teorico
% e' h_ott ~ eps_mach^(1/4) ~ 1.2e-4, cioe' k=4: infatti verificato
% numericamente che l'errore e' minimo (~1e-7) a k=4 e cresce fino a
% valori enormi (~40) a k=8. Questo va discusso nel report come limite
% teorico noto delle differenze finite del second'ordine, non nascosto.
 x = x(:);
    n = length(x);
    h = steps_fd(x, k, mode);

    phi0 = trig16_local_terms(x, n);
    phip = trig16_local_terms(x + h, n);
    phim = trig16_local_terms(x - h, n);

    g = (phip - phim) ./ (2*h);           % differenza centrata: g_i = phi_i'(x_i)
    d = (phip - 2*phi0 + phim) ./ (h.^2); % differenza seconda centrata: d_i = phi_i''(x_i)

    H = spdiags(d, 0, n, n);
end

function phi = trig16_local_terms(x, n)
% Restituisce il vettore [phi_1(x_1), ..., phi_n(x_n)], la decomposizione
% "per variabile" di F(x) descritta sopra.
    i = (1:n)';
    c = 2*ones(n,1);
    c(n) = -(n-1);
    phi = i.*(1-cos(x)) + c.*sin(x);
end