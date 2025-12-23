function [xk,fk,gradfk_norm,k,xseq,btseq,tau_new,alphas,pks] = ...
    modified_newton_method2(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

% Armijo condition
farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;

% max number of attempts for tau correction
maxtau = 100;

% Allocations
n = length(x0);
xseq   = zeros(n, kmax);
btseq  = zeros(1, kmax);
alphas = zeros(1, kmax);
pks    = zeros(n, kmax);
tau_new = zeros(maxtau+1, kmax);

% Initialization
xk = x0;
fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);
k = 0;

% Main loop
while k < kmax && gradfk_norm >= tolgrad

    % Hessian at current point
    Hk = hessf(xk);

    % Try Cholesky on the original Hessian
    [R,flag] = chol(Hk);

    % Initial tau
    if flag == 0
        tau_k = 0;          % Hessian already SPD
    else
        tau_k = beta;      % start correction
    end

    tauk = zeros(maxtau+1,1);
    tauk(1) = tau_k;

    % Modified Hessian until SPD
    for j = 1:maxtau
        if n > 1e-4
            Bk = Hk+tauk(j)*speye(n); 
        else
            Bk = Hk+tauk(j)*eye(n); 
        end
        [R,flag] = chol(Bk);
        if flag == 0
            break
        else
            tauk(j+1) = max(beta, 2*tauk(j));
        end
    end

    % Modified Newton direction
    pk = R \ (R' \ (-gradfk));

    % Backtracking line search (Armijo)
    alpha = 1;
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;

    xnew = xk + alpha * pk;
    fnew = f(xnew);

    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end

    % Safeguard
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        disp("Backtracking: massimo numero di iterazioni raggiunto");
        break;
    end

    % Update
    k = k + 1;
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);

    % Store results
    xseq(:,k)   = xk;
    btseq(k)    = bt;
    alphas(k)   = alpha;
    pks(:,k)    = pk;
    tau_new(1:length(tauk),k) = tauk;

end

% Trim outputs
xseq   = xseq(:,1:k);
btseq  = btseq(1:k);
alphas = alphas(1:k);
pks    = pks(:,1:k);

% Add x0 as first point (useful for plots)
xseq = [x0, xseq];

end
