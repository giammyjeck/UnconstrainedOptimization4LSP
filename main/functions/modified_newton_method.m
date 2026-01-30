function [xk,fk,gradfk_norm,k,xseq,btseq,alphas,gradfk_seq,fk_seq,tau_new,pks] = modified_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

% MODIFIED_NEWTON_METHOD  Modified Newton method with Hessian correction
%
%   [xk, fk, gradfk_norm, k, xseq, btseq, tau_new,alphas,pks] = ...
%       modified_newton_method(x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta)
%
%       This functions is aimed to solve large scale numerical optimization
%       problems using the modified Newton method with backtracking techniques.
%
%   INPUT ARGUMENTS:
%       x0              : initial point (column vector)
%       f               : function handle for the scalar objective function f(x)
%       gradf           : function handle for the gradient gradf(x)
%       hessf           : function handle for the Hessian hessf(x)
%       kmax            : maximum number of iterations
%       tolgrad         : tolerance on the gradient norm for the stopping condition
%       c1, rho, btmax  : parameters for Armijo/backtracking (0 < c1 < 1, 0 < rho < 1)
%       beta            : minimum initial increment for the Hessian correction (beta > 0)
%
%   OUTPUT ARGUMENTS:
%       xk, fk, gradfk_norm  : final point, objective value, gradient norm at the solution
%       k                    : number of iterations performed
%       xseq                 : sequence of iterates [x0, x1, ..., xk]  (n × (k+1))
%       btseq                : number of backtracking steps at each iteration (1 × k)
%       tau_new              : (maxit+1) × k matrix storing the tau values used at each iteration
%       alphas               : step lengths
%       pks                  : search directions


% INPUT CHECKS
if ~isnumeric(x0) || ~isvector(x0)
    error('x0 must be a numeric column vector.');
end
xk = x0(:); % Forcing column vector.
n = length(xk);

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

if ~isscalar(kmax) || kmax <= 0 || floor(kmax) ~= kmax
    error('kmax must be a positive integer.');
end

if ~isscalar(tolgrad) || tolgrad <= 0
    error('tolgrad must be a positive scalar.');
end

if ~isscalar(c1) || c1 <= 0 || c1 >= 1
    error('c1 must satisfy 0 < c1 < 1.');
end

if ~isscalar(rho) || rho <= 0 || rho >= 1
    error('rho must satisfy 0 < rho < 1.');
end

if ~isscalar(btmax) || btmax <= 0 || floor(btmax) ~= btmax
    error('btmax must be a positive integer.');
end

if ~isscalar(beta) || beta <= 0
    error('beta must be a positive scalar.');
end

gradfk_norm = norm(gradfk);
if gradfk_norm < tolgrad
    warning('Initial point already satisfies the stopping criterion.');
end


% Function handle for the armijo condition.
farmijo = @(fk, alpha, c1_gradfk_pk) ...
    fk + alpha * c1_gradfk_pk;


% Inner parameters.
tol = 1e-6;     % Tolerance to check the positivness of Hk diagonal.
maxit = 200;    % Max number of iteration to check the positivness of Hk diagonal.

% Preallocations.
xseq =          zeros(length(x0), kmax); 
btseq =         zeros(1, kmax);         
alphas =        zeros(1, kmax);
gradfk_seq =    zeros(1, kmax);
fk_seq =        zeros(1, kmax);
pks =           zeros(length(x0),kmax);
tau_new =       zeros(maxit+1,kmax); 


k = 0;
while k < kmax && gradfk_norm >= tolgrad
    
    % Compute the hessian matrix.
    Hk = hessf(xk);
    
    % Check if the hessian matrix is positive definite.
    diagHk = full(diag(Hk));
    isPositive = all(diagHk>tol);

    if isPositive == true
        tau_k = 0; % No need to add a correction term 
    else
        tau_k = beta-min(diagHk); 
    end
    
    % Vector to store the history of the correction term per iteration.
    tauk = zeros(maxit+1,1); 
    tauk(1)= tau_k;

    % Once the parameter tau is found then the correction of the hessian 
    % can be built.
    % In order to check if the corrected matrix is positive definite, we 
    % attempt to perform the incomplete choleski factorization: 
    % if Bk is not positive definite, then we get an error and consequently
    % we must try to improve it.
    for j = 1:maxit

            if n > 1e-4
                Bk = Hk+tauk(j)*speye(n); 
            else
                Bk = Hk+tauk(j)*eye(n); 
            end
            
            % Cholesky factorization.
            [R,flag] = chol(Bk);

            % Chech if the correction is good enough (Is Bk positive
            % definite?).
            if flag == 0
                %fprintf('Bk is positive definite k: %d iteration j: %d\n', k, j);
                break
            else
                % If the Bk in not positive definite, then we need to correct it.
                tauk(j+1) = max(beta, 2*tauk(j)); 
                %fprintf('Bk is NOT positive definite k: %d iteration j: %d\n', k, j);
            end
    end

    % Once we obtain a matrix Bk which is positive definite we solve the
    % following system with a direct solver.
    pk = R\(R'\(-gradfk));
    


    % BACKTRACKING.
    
    alpha = 1; % Reset the value of alpha (parameter used for linesearch).
    xnew = xk + alpha * pk;   % Computing the candidate solution at step k.
    fnew = f(xnew);
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;

    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk) 

        alpha = rho * alpha;  % Step reduction. 
        xnew = xk + alpha * pk; 
        fnew = f(xnew);
        bt = bt + 1;
    end


    % Check if the maximum number of backtracking iterations is reached.
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        disp('Maximum backtracking iterations reached, stopping.');
        k = k+1;
        xseq(:, k)      = xk;
        btseq(k)        = bt;
        tau_new(:, k)   = tauk; 
        alphas(k)       = alpha;
        pks(:, k)       = pk;
        gradfk_seq(k)   = gradfk_norm;
        fk_seq(k)       = fk;
        break;
    end
    
    % Update variables.
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    

    k = k + 1; 

    % Storing.
    xseq(:, k)      = xk;
    btseq(k)        = bt;
    tau_new(:, k)   = tauk; 
    alphas(k)       = alpha;
    pks(:, k)       = pk;
    gradfk_seq(k)   = gradfk_norm;
    fk_seq(k)        = fk;

end %while loop on k


% Trimming the final structures.
xseq            = xseq(:, 1:k);
btseq           = btseq(1:k);
alphas          = alphas(1:k);
pks             = pks(:, 1:k);
gradfk_seq      = gradfk_seq(1:k);
fk_seq          = fk_seq(1:k);
xseq = [x0, xseq];


end

