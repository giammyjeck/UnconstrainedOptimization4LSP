function [xk,fk,gradfk_norm,k,xseq,btseq,alphas,gradfk_seq,fk_seq,tau_new,pks] = ...
    modified_newton_method3(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

% MODIFIED_NEWTON_METHOD3
% Modified Newton with Hessian correction, backtracking, and tau-retry
%
% OUTPUT format aligned with modified_newton_method:
% xk, fk, gradfk_norm, k, xseq, btseq, alphas, gradfk_seq, fk_seq, tau_new, pks

% Parameters
maxtau = 100;
max_retries = 5;

n = length(x0);

% Preallocations
xseq = zeros(n, kmax);
btseq = zeros(1, kmax);
alphas = zeros(1, kmax);
gradfk_seq = zeros(1, kmax);
fk_seq = zeros(1, kmax);
pks = zeros(n, kmax);
tau_new = zeros(maxtau+1, kmax); % store tau history per iteration

% Initialization
xk = x0(:);
fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);
k = 0;

while k < kmax && gradfk_norm >= tolgrad
    
    Hk = hessf(xk);
    
    % Initial tau estimate
    diagH = diag(Hk);
    if all(diagH > 0)
        tau = 0;
    else
        tau = max(0, -min(diagH) + beta);
        if ~isfinite(tau), tau = beta; end
    end
    
    % Make Hessian SPD
    tauk = zeros(maxtau+1,1);
    tauk(1) = tau;
    for j = 1:maxtau
        Bk = Hk + tau*speye(n);
        [R,flag] = chol(Bk);
        if flag == 0
            break
        else
            tau = max(beta, 2*tau);
            tauk(j+1) = tau;
        end
    end
    
    if flag ~= 0
        warning('Hessian correction failed: Bk not SPD.');
        break;
    end
    
    % Modified Newton direction
    pk = R \ (R' \ (-gradfk));
    
    % Ensure descent
    gTp = gradfk' * pk;
    if gTp >= 0
        pk = -gradfk;
        gTp = -gradfk_norm^2;
    end
    
    % Backtracking line search
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
    
    % Tau-retry if backtracking fails
    retry = 0;
    while bt == btmax && fnew > fk + c1*alpha*gTp && retry < max_retries
        retry = retry + 1;
        tau = max(beta, 10*tau);
        Bk = Hk + tau*speye(n);
        [R2,flag2] = chol(Bk);
        if flag2 ~= 0
            continue
        end
        pk = R2 \ (R2' \ (-gradfk));
        gTp = gradfk' * pk;
        if gTp >= 0
            pk = -gradfk;
            gTp = -gradfk_norm^2;
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
        warning('Backtracking failed even after tau-retry.');
        break;
    end
    
    % Update iterate
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    k = k + 1;
    
    % Store results
    xseq(:,k) = xk;
    btseq(k) = bt;
    alphas(k) = alpha;
    gradfk_seq(k) = gradfk_norm;
    fk_seq(k) = fk;
    tau_new(:,k) = tauk;
    pks(:,k) = pk;
end

% Trim arrays
xseq = xseq(:,1:k);
btseq = btseq(1:k);
alphas = alphas(1:k);
gradfk_seq = gradfk_seq(1:k);
fk_seq = fk_seq(1:k);
pks = pks(:,1:k);
tau_new = tau_new(:,1:k);

% Prepend x0 to xseq
xseq = [x0, xseq];

end
