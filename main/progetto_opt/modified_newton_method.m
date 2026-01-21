function [xk,fk,gn,k,xseq,btseq,alphas,gnseq,fseq,tau_hist] = ...
    modified_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta,store_seq)
% MODIFIED_NEWTON_METHOD
% ------------------------------------------------------------
% Modified Newton con:
%  - correzione Hessiana: Bk = Hk + tau*I finché SPD (Cholesky ok)
%  - direzione: pk = -Bk^{-1} gk (risolta con Cholesky)
%  - backtracking line search (Armijo)
%  - tau-retry: se backtracking fallisce, aumento tau e riprovo (robustezza)
%
% INPUT:
%   x0        : punto iniziale (n×1)
%   f         : handle f(x)
%   gradf     : handle grad f(x)
%   hessf     : handle Hess f(x) (meglio sparse)
%   kmax      : max iterazioni
%   tolgrad   : tolleranza su ||grad||
%   c1,rho    : parametri Armijo/backtracking
%   btmax     : max iterazioni backtracking
%   beta      : minimo shift tau
%   store_seq : true -> salva xseq (consigliato solo per n=2)
%
% OUTPUT:
%   xk,fk,gn,k
%   xseq      : sequenza dei punti (solo se store_seq=true), include x0 come 1ª colonna
%   btseq     : # backtracking a ogni iterazione
%   alphas    : alpha scelti
%   gnseq     : ||grad|| a ogni iterazione
%   fseq      : f(xk) a ogni iterazione
%   tau_hist  : tau usato a ogni iterazione

% Parametri interni
maxtau = 100;        % max tentativi per rendere SPD
max_retries = 5;     % max tentativi di tau-retry se backtracking fallisce

n = length(x0);
if nargin < 11 || isempty(store_seq)
    store_seq = (n == 2);
end

% Allocazioni leggere
btseq    = zeros(1, kmax);
alphas   = zeros(1, kmax);
gnseq    = zeros(1, kmax);
fseq     = zeros(1, kmax);
tau_hist = zeros(1, kmax);

% Sequenza punti: solo se serve (per n grandi esplode la memoria)
if store_seq
    xseq = zeros(n, kmax);
else
    xseq = [];
end

% Inizializzazione
xk = x0;
fk = f(xk);
gk = gradf(xk);
gn = norm(gk);
k  = 0;

while k < kmax && gn >= tolgrad

    Hk = hessf(xk);

    % Provo chol su Hk: se fallisce, non SPD
    [~,flag] = chol(Hk);

    % Tau iniziale "furbo" (usa la diagonale)
    if flag == 0
        tau = 0;
    else
        dH = diag(Hk);
        tau = max(0, -min(dH) + beta);
        if ~isfinite(tau), tau = beta; end
    end

    % Rendo SPD: aumento tau finché chol ok
    for j = 1:maxtau
        Bk = Hk + tau * speye(n);
        [R,flag] = chol(Bk);
        if flag == 0
            break
        else
            tau = max(beta, 2*tau);
        end
    end

    if flag ~= 0
        warning("Correzione tau fallita: Hessiana non resa SPD.");
        break;
    end

    % Direzione Newton modificata
    pk = R \ (R' \ (-gk));

    % Check direzione di discesa
    gTp = gk' * pk;
    if gTp >= 0
        pk  = -gk;
        gTp = -gn^2;
    end

    % Backtracking Armijo
    alpha = 1;
    bt = 0;
    xnew = xk + alpha*pk;
    fnew = f(xnew);

    while bt < btmax && fnew > fk + c1*alpha*gTp
        alpha = rho*alpha;
        xnew = xk + alpha*pk;
        fnew = f(xnew);
        bt = bt + 1;
    end

    % ---- tau-retry se backtracking fallisce ----
    retry = 0;
    while bt == btmax && fnew > fk + c1*alpha*gTp && retry < max_retries
        retry = retry + 1;

        tau = max(beta, 10*tau);          % aumento più aggressivo
        Bk = Hk + tau * speye(n);
        [R2,flag2] = chol(Bk);
        if flag2 ~= 0
            continue
        end

        pk = R2 \ (R2' \ (-gk));
        gTp = gk' * pk;
        if gTp >= 0
            pk  = -gk;
            gTp = -gn^2;
        end

        alpha = 1;
        bt = 0;
        xnew = xk + alpha*pk;
        fnew = f(xnew);

        while bt < btmax && fnew > fk + c1*alpha*gTp
            alpha = rho*alpha;
            xnew = xk + alpha*pk;
            fnew = f(xnew);
            bt = bt + 1;
        end
    end

    if bt == btmax && fnew > fk + c1*alpha*gTp
        warning("Backtracking fallito anche dopo tau-retry: stop.");
        break;
    end
    % ------------------------------------------

    % Aggiornamento
    k = k + 1;
    xk = xnew;
    fk = fnew;
    gk = gradf(xk);
    gn = norm(gk);

    % Salvataggi
    btseq(k)    = bt;
    alphas(k)   = alpha;
    gnseq(k)    = gn;
    fseq(k)     = fk;
    tau_hist(k) = tau;

    if store_seq
        xseq(:,k) = xk;
    end
end

% Taglio a lunghezza effettiva
btseq    = btseq(1:k);
alphas   = alphas(1:k);
gnseq    = gnseq(1:k);
fseq     = fseq(1:k);
tau_hist = tau_hist(1:k);

% Aggiungo x0 davanti (per plot)
if store_seq
    xseq = [x0, xseq(:,1:k)];
else
    xseq = [];
end

end
