function [xk,fk,gradfk_norm,k,xseq,btseq,tau_new,alphas,pks] = modified_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

% MODIFIED_NEWTON_METHOD  Modified Newton method with Hessian correction
% [xk, fk, gradfk_norm, k, xseq, btseq, tau_new,alphas,pks] = ...
%     modified_newton_method(x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta)
%
%   This functions is aimed to solve large scale numerical optimization
%   problems using the modified Newton method with backtracking techniques.
%
% INPUT:
%   x0      : initial point (column vector)
%   f       : function handle for the scalar objective function f(x)
%   gradf   : function handle for the gradient gradf(x)
%   hessf   : function handle for the Hessian hessf(x)
%   kmax    : maximum number of iterations
%   tolgrad : tolerance on the gradient norm for the stopping condition
%   c1, rho, btmax : parameters for Armijo/backtracking (0 < c1 < 1, 0 < rho < 1)
%   beta    : minimum initial increment for the Hessian correction (beta > 0)
%
% OUTPUT:
%   xk, fk, gradfk_norm : final point, objective value, gradient norm at the solution
%   k        : number of iterations performed
%   xseq     : sequence of iterates [x0, x1, ..., xk]  (n × (k+1))
%   btseq    : number of backtracking steps at each iteration (1 × k)
%   tau_new  : (maxit+1) × k matrix storing the tau values used at each iteration
%   alphas   :
%   pks      :


% INPUT CHECKS

% x0
if ~isnumeric(x0) || ~isvector(x0)
    error('x0 must be a numeric column vector.');
end
xk = x0(:); % force column vector
n = length(xk);

% f
if ~isa(f,'function_handle')
    error('f must be a function handle.');
end
try
    fk = f(xk);
catch
    error('f(x0) cannot be evaluated.');
end
if ~isscalar(fk) || ~isreal(fk)
    error('f(x) must return a real scalar.');
end

% gradf
if ~isa(gradf,'function_handle')
    error('gradf must be a function handle.');
end
try
    gradfk = gradf(xk);
catch
    error('gradf(x0) cannot be evaluated.');
end
if ~isnumeric(gradfk) || ~isequal(size(gradfk),[n,1])
    error('gradf(x) must return a column vector of the same size as x.');
end

% hessf
if ~isa(hessf,'function_handle')
    error('hessf must be a function handle.');
end
try
    H0 = hessf(xk);
catch
    error('hessf(x0) cannot be evaluated.');
end
if ~isnumeric(H0) || ~isequal(size(H0),[n,n])
    error('hessf(x) must return an n-by-n matrix.');
end

% kmax
if ~isscalar(kmax) || kmax <= 0 || floor(kmax) ~= kmax
    error('kmax must be a positive integer.');
end

% tolgrad
if ~isscalar(tolgrad) || tolgrad <= 0
    error('tolgrad must be a positive scalar.');
end

% Armijo / backtracking parameters
if ~isscalar(c1) || c1 <= 0 || c1 >= 1
    error('c1 must satisfy 0 < c1 < 1.');
end

if ~isscalar(rho) || rho <= 0 || rho >= 1
    error('rho must satisfy 0 < rho < 1.');
end

if ~isscalar(btmax) || btmax <= 0 || floor(btmax) ~= btmax
    error('btmax must be a positive integer.');
end

% beta (Hessian correction parameter)
if ~isscalar(beta) || beta <= 0
    error('beta must be a positive scalar.');
end

% Consistency check: gradient norm already below tolerance
gradfk_norm = norm(gradfk);
if gradfk_norm < tolgrad
    warning('Initial point already satisfies the stopping criterion.');
end


% Function handle for the armijo condition
farmijo = @(fk, alpha, c1_gradfk_pk) ...
    fk + alpha * c1_gradfk_pk;


% Initializations

% !!! i parametri sotto dovrebbero essere settati con coerenza
% quindi controllare e definire la coerenza
tol = 1e-6;     % tol to check the positivness of Hk diagonal
maxit = 100;    % max number of iteration to check the positivness of Hk diagonal


xseq = zeros(length(x0), kmax); % matrix to store computed solution 
btseq = zeros(1, kmax);         % vector to store number of backtracking iteration  
alphas = zeros(1, kmax);
pks = zeros(length(x0),kmax);
tau_new = zeros(maxit+1,kmax); % matrix to save the values of tau_k


k = 0;
while k < kmax && gradfk_norm >= tolgrad
    
    % Compute the hessian matrix
    % !!! poi qui useremo le differenze finite
    Hk = hessf(xk);
    
    % Check if the hessian matrix is positive definite
    % !!! only necessary not sufficient condition (implement other)
    diagHk = diag(Hk);
    isPositive = all(diagHk>tol);

    if isPositive == true
        tau_k = 0; % no need to add a correctional term 
    else
        tau_k = beta-min(diagHk); % !!! check se va messo max(0,)
        % if min(diagHK) is negative ? 
    end
    
    %Bk = Hk+tau_k*eye(size(Hk));
    tauk = zeros(maxit+1,1); % vector to store the history of correction term per iteration
    tauk(1)= tau_k;

    % Once the parameter tau is found then the correction of the hessian 
    % can be built.
    % In order to check if the corrected matrix is positive definite, we 
    % attempt to perform the incomplete choleski factorization: 
    % if Bk is not positive definite, then you get an error and try improve it.
    for j = 1:maxit
            Bk = Hk+tauk(j)*eye(size(Hk)); % !!! dovremmo controllare che Bk sia simmetrica prima di usare chol?
            
            [R,flag] = chol(Bk);

            % Chech if the correction is good enough (Is Bk positive definite?)
            if flag == 0
                fprintf('Bk is positive definite k: %d iteration j: %d\n', k, j);
                break
            else
                % If the Bk in not positive definite, then we need to correct it.
                % new value of tau_k !!! perché proprio questo valore?
                tauk(j+1) = max(beta, 2*tauk(j)); % controllare che tau non possa essere negativo?
                fprintf('Bk is NOT positive definite k: %d iteration j: %d\n', k, j);
            end
    end

    % Once we obtain a matrix Bk which is positive definite we can solve the
    % following system with a direct solver 
    %!!! pk = pcg(Bk, -gradfk, [], [], R, R');
    pk = R\(R'\(-gradfk));
    
    fprintf('gradf''*p_k = %.3e\n', gradfk' * pk);
    % NOTE: there's no need to check if pk is a descent direction because
    % of remark2 !!! CONTROLLARE SU TEORIA PERCHEé
    
    
    % Reset the value of alpha
    % !!! for teoretical properties you need alpha to be 1
    alpha = 1; % this is the parameter used for linesearch
    
    % Compute the candidate new xk
    xnew = xk + alpha * pk;
    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;

    % Backtracking strategy from here...: 
    % 2nd condition is the Armijo condition not satisfied
    fprintf('fnew - farmijo = %.3e\n', fnew - farmijo(fk, alpha, c1_gradfk_pk));

    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk) 

        % Riduci alpha
        alpha = rho * alpha;  
        fprintf('alpha = %.3e at bt = %.1f\n', alpha, bt);
    
        % Aggiorna xnew e fnew
        xnew = xk + alpha * pk;
        fnew = f(xnew);
    
        % Incrementa contatore
        bt = bt + 1;
    end


    % Check if the maximum number of backtracking iterations is reached
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        disp('Maximum backtracking iterations reached, stopping.');
        break;
    end
    % ...to here
    
    % Update xk, fk, gradfk_norm
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    

    % Increase the step by one
    k = k + 1; % !!! CONTROLLARE se è  più robusto incrementare dopo aver salvato

    % Storing
    xseq(:, k)      = xk;
    btseq(k)        = bt;
    tau_new(:, k)   = tauk; %Store current tauk values
    alphas(k)    = alpha;
    pks(:, k)    = pk;

end


% "Cut" xseq and btseq to the correct size
xseq   = xseq(:, 1:k);
btseq  = btseq(1:k);
alphas = alphas(1:k);
pks    = pks(:, 1:k);

% "Add" x0 at the beginning of xseq (otherwise the first el. is x1)
xseq = [x0, xseq];


end


% TO DO:

% INPUT
%     : controllare e scrivere la dimensione e il tipo expected di ogni parametro
%     : definire dei controlli di coerenza per i parametri in modo da 
%       non andare a lavorare nell'ordine della precisione di macchina
% DONE: aggiungere dei controlli generici sugli input

% ALTRO
%     : (diagHk = diag(Hk); isPositive = all(diagHk>tol);) E' condizione necessaria ma non sufficiente! 
%%% CONDIZIONE SUFFICIENTE dominanza diagonale (costo O(N^2))
%%% Cerchi di gerghscorins
%%% metodi di lanczos
%       eventualmente provare altre condizioni Necessarie purché veloci da
%       calcolare, poi nel report confrontiamo che succede
%     : (tau_k = beta-min(diagHk);) controllare nella teoria se max(0,...)
%     : (Bk = Hk+tauk(j)*eye(size(Hk));) dovremmo controllare che sia
%       simemtrica prima di usare chol? cioè un controllo basic ulteriore? se non lo
%       fosse ? continuiamo a migliorarla ? 
%     : (tauk(j+1) = max(beta, 2*tauk(j))); quando aggiorniamo Bk
%       correggendo il tauk, facciamo questa correzzione, ma perché ? controlalre teoria e
%       eventualmente proporre altre soluzioni
%     : pk è descent direction ? NOTE: there's no need to check if pk is a descent direction because
%       of remark2 !!! CONTROLLARE SU TEORIA PERCHEé
%     : linesearch !!! for teoretical properties you need alpha to be 1
%     
%     : aggiungere altri modi di calcolare la Bk guardando nel book di
%       teoria

%BACKTRACKING
%
%     : rho dalla teoria del backtracking sappiamo che può essere fisso o chosen
%       by interpolation rho \in [\rho_l, \rho_u]
%     : Controllare CON quale valore alpha viene soddisfatta armijo perché se alpha <<1
%       abbiamo stagnation... 
%     : ATTENZIONE: Non fare l'errore di permettere che alpha sia < eps, bisogna
%       stare attenti a scegliere un rho e btmax che non permetta alpha<eps,
%       NEPPURE TEORICAMENTE, cioè quando fissiamo i parametri cerchiamo delle
%       condizioni che le legano... rho = 0.5 e btmax  = 50 è un errore
%       GRAVESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
%       anche btmax = 30 nonè giusto perché vado lento
%       con 50 backtracking e rho = 0.5 raggiungiamo eps... sbagliato
%
% working : AGGIUNGERE CONDIZIone di curvatura, MA soprattutto Ha senso
%           farlo, se si , quando  