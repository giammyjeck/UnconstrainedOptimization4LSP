function [xk,fk,gradfk_norm,k,xseq,btseq,pks,inner_iters] = truncated_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,max_cg) 
% TRUNCATED_NEWTON_METHOD Solves unconstrained optimization problems.
%   This function implements the Truncated Newton method (also known as 
%   Newton-CG). It uses a Conjugate Gradient (CG) inner loop to compute 
%   the descent direction and an Armijo-backtracking line search for 
%   global convergence.
%
%   [xk, fk, gradfk_norm, k, xseq, btseq, pks, inner_iters] = ...
%       TRUNCATED_NEWTON_METHOD(x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, max_cg)
%
%   INPUT ARGUMENTS:
%       x0          - Initial guess (column vector)
%       f           - Function handle for f(x)
%       gradf       - Function handle for the gradient gradf(x)
%       hessf       - Function handle for the Hessian matrix H(x)
%       kmax        - Maximum number of outer (Newton) iterations
%       tolgrad     - Stopping tolerance on the norm of the gradient
%       c1          - Armijo condition parameter (0 < c1 < 1)
%       rho         - Backtracking reduction factor (0 < rho < 1)
%       btmax       - Maximum number of backtracking steps per iteration
%       max_cg      - Maximum number of inner (CG) iterations
%
%   OUTPUT ARGUMENTS:
%       xk          - Final point reached by the algorithm
%       fk          - Function value at xk
%       gradfk_norm - Norm of the gradient at xk
%       k           - Total number of outer iterations performed
%       xseq        - Sequence of points generated (including x0)
%       btseq       - Number of backtracking steps per iteration
%       pks         - Sequence of descent directions computed
%       inner_iters - Number of CG iterations performed at each step k



% INPUT CHECKS

if ~isnumeric(x0) || ~isvector(x0)
    error('x0 must be a numeric column vector.');
end
xk = x0(:); % force column vector
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

if ~isscalar(max_cg) || max_cg <= 0
    error('max_cg must be a positive scalar.');
end

gradfk_norm = norm(gradfk);
if gradfk_norm < tolgrad
    warning('Initial point already satisfies the stopping criterion.');
end


% Function handle for the armijo condition
farmijo = @(fk, alpha, c1_gradfk_pk) ...
    fk + alpha * c1_gradfk_pk;

% Variables iniziatization
xseq        = zeros(length(x0), kmax);
btseq       = zeros(1, kmax);
alphas      = zeros(1, kmax);
pks         = zeros(length(x0), kmax);
inner_iters = zeros(1, kmax); 


k = 1;
%k = 0;
i = 0;
while k <= kmax && gradfk_norm >= tolgrad
    i= i+1;

    % The system we need to solve is Hess(fk)*pk = -graf(fk) <-> Hk*z=ck
    z = zeros(length(x0),1); 
    Hk = hessf(xk);
    ck = -gradfk;

    % Initialize p_tn with the steepest descent direction (-gradfk) 
    % as a fallback to guarantee a valid descent direction even if 
    % the CG inner loop fails or terminates at the first iteration.
    p_tn = ck; 

    
    eta_k = min(0.5, sqrt(gradfk_norm));

    r = ck - Hk * z; % Residual of the system
    r_old = r'*r;
    dk = r; % d is the conjugate direction, at the first iteration z = 0 so dk = ck.

    stop_inner = false; % Boolean variable used to understand whether the inner loop got to convergence,
    %  it has to go back to false at each iteration k
    
    j = 0;
    while ~stop_inner && j < max_cg % Inner loop for solving the system with CG

        Hdk = Hk*dk;
        curv = dk'*Hdk;
        
        %if curv > 0 % If the curvature is positive we can proceed with CG method.
        if curv <= 1e-12
            
            if j == 0
                p_tn = -gradfk;
                
            else
                p_tn = z;
            end       
            stop_inner = true;
            break

        else

            alpha_j = r_old/curv;
            z = z + alpha_j * dk;
            r = r - alpha_j * Hdk; 

            
            % Update for the next iteration
            rk_new = r'*r;
            beta_j = rk_new/r_old;
            dk = r + beta_j * dk;
            r_old = rk_new; 
            p_tn = z;

            j = j+1;

        end

        % Check on convergence
        if norm(r)/norm(ck) <= eta_k 
            p_tn = z;      
            stop_inner = true;
            break;
        end

    end

    inner_iters(k) = j;
    pk = p_tn;
    
    if norm(pk) <= 1e-12
        disp('Truncated Newton: null direction, stop.');
        if gradfk_norm < tolgrad
             return;
        else
            warning('Truncated Newton: pk near zero but gradient not converged.');
        end

    end


    % Armijo Backtracking

    alpha = 1;
    xnew = xk + alpha * pk;
    fnew = f(xnew);
    c1_gradfk_pk = c1 * (gradfk' * pk);

    bt = 0;
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha; % step reduction
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end

    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        disp('Backtracking (Truncated Newton): maximum number of iterations reached.');
        k = k+1;
        xseq(:, k) = xk;
        btseq(k) = bt;
        pks(:, k) = pk;
        alphas(k) = alpha;
        break;            
    end
            
    % Update variables
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    k = k + 1;
    
    % Storage
    xseq(:, k) = xk;
    btseq(k) = bt;
    pks(:, k) = pk;
    alphas(k) = alpha;

end %while loop on k

% Trim outputs
xseq   = xseq(:, 1:k);
btseq  = btseq(1:k);
pks    = pks(:, 1:k);

xseq = [x0, xseq]; 

end %function end