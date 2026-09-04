function [xk,fk,gradfk_norm,k,xseq,btseq,alphas,gradfk_seq,fk_seq,n_flips] = modified_newton_method_flip(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax)
 
% MODIFIED_NEWTON_METHOD_FLIP
%
% Modified Newton con correzione della Hessiana ESCLUSIVAMENTE tramite
% flipping degli autovalori negativi (Bk ha gli stessi autovettori di
% Hk, con |lambda_i| al posto di lambda_i dove lambda_i <= 0), invece
% di Hk + tau*I.
%
% Nessun fallback: se Hk non e' diagonale, si esegue SEMPRE la
% decomposizione spettrale completa (eig), qualunque sia n. Per n
% grande questo e' costoso (vedi discussione teorica a parte) -- e'
% una scelta voluta per questo esperimento, non un'implementazione
% pensata per produzione su problemi grandi.
%
% Se Hk e' diagonale (es. trig16), il flipping equivale a flippare
% direttamente le entrate della diagonale: stessa operazione, ma senza
% bisogno di chiamare eig() ne' di materializzare una matrice densa,
% perche' per una matrice diagonale gli autovalori SONO la diagonale e
% gli autovettori sono la base canonica.
%
% OUTPUT aggiuntivo rispetto a modified_newton_method.m:
%   n_flips : numero di iterazioni in cui almeno un autovalore e'
%             stato flippato
 
if ~isnumeric(x0) || ~isvector(x0)
    error('x0 must be a numeric column vector.');
end
xk = x0(:);
n = length(xk);
 
fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);
 
farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;
 
eps_floor = 1e-8;   % "epsilon" del libro: poco sopra la precisione di macchina
 
xseq         = zeros(n, kmax);
btseq        = zeros(1, kmax);
alphas       = zeros(1, kmax);
gradfk_seq   = zeros(1, kmax);
fk_seq       = zeros(1, kmax);
n_flips      = 0;
toldiag = 1e-6; 
k = 0;
while k < kmax && gradfk_norm >= tolgrad
 
    Hk = hessf(xk);
 
    is_diag = nnz(Hk - spdiags(spdiags(Hk,0), 0, n, n)) == 0;
 
    if is_diag
        d = full(diag(Hk));
        applied = any(d <= toldiag);
        d_mod = d;
        d_mod(d <= toldiag) = max(abs(d(d <= toldiag)), eps_floor);
        Bk = spdiags(d_mod, 0, n, n);
    else
        % Decomposizione spettrale completa, SEMPRE, qualunque sia n.
        [Q, D] = eig(full(Hk));
        lambda = diag(D);
        applied = any(lambda <= 0);
        lambda_mod = lambda;
        lambda_mod(lambda <= 0) = max(abs(lambda(lambda <= 0)), eps_floor);
        Bk = Q * diag(lambda_mod) * Q';
    end
 
    if applied
        n_flips = n_flips + 1;
    end
 
    pk = -Bk \ gradfk;
 
    % BACKTRACKING
    alpha = 1;
    xnew = xk + alpha * pk;
    fnew = f(xnew);
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;
 
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end
 
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        k = k+1;
        xseq(:, k)      = xk;
        btseq(k)        = bt;
        alphas(k)       = alpha;
        gradfk_seq(k)   = gradfk_norm;
        fk_seq(k)       = fk;
        break;
    end
 
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
 
    k = k + 1;
 
    xseq(:, k)      = xk;
    btseq(k)        = bt;
    alphas(k)       = alpha;
    gradfk_seq(k)   = gradfk_norm;
    fk_seq(k)       = fk;
 
end
 
xseq            = xseq(:, 1:k);
btseq           = btseq(1:k);
alphas          = alphas(1:k);
gradfk_seq      = gradfk_seq(1:k);
fk_seq          = fk_seq(1:k);
xseq = [x0, xseq];
 
end
 