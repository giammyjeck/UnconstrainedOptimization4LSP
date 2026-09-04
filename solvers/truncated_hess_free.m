function [xk,fk,gradfk_norm,k,xseq,btseq,pks,inner_iters] = truncated_hess_free(x0,f,gradf,kmax,tolgrad,c1,rho,btmax,max_cg,h_fd) 

% TRUNCATED_NEWTON_METHOD Truncated Newton method, matrix-free.
%
%   This function solves unconstrained numerical optimization problems 
%   using a Truncated Newton approach. 
%   The Hessian is never built explicitly: at each outer iteration the
%   Newton system H(xk)*p = -gradf(xk) is solved approximately with the
%   Conjugate Gradient (CG) method, and every Hessian-vector product
%   H(xk)*p required by CG is approximated with a centered finite
%   difference on the gradient.
%
%   INPUT ARGUMENTS:
%       x0              : Initial point (column vector)
%       f               : Function handle for the objective function f(x)
%       gradf           : Function handle for the gradient gradf(x)
%       kmax            : Maximum number of outer iterations
%       tolgrad         : Stopping tolerance on the gradient norm for the stopping criterion
%       c1, rho, btmax  : parameters for Armijo/backtracking (0 < c1 < 1, 0 < rho < 1)
%       max_cg          : Maximum number of inner (CG) iterations
%	h_fd            : Finite-difference step for the Hessian-vector product
%
%   OUTPUT ARGUMENTS:
%       xk, fk, gradfk_norm  : final point, objective value and gradient norm at the solution
%       k                    : Total number of iterations performed
%       xseq                 : Sequence of points generated (n x (k+1))
%       btseq                : Number of backtracking steps per iteration (1 x k)
%       pks                  : Sequence of search directions 
%       inner_iters          : Number of CG iterations performed per step 

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

if nargin < 10 || isempty(h_fd)
    h_fd = eps^(1/3);
end

gradfk_norm = norm(gradfk);
if gradfk_norm < tolgrad
    warning('Initial point already satisfies the stopping criterion.');
end


% Function handle for the armijo condition.
farmijo = @(fk, alpha, c1_gradfk_pk) ...
    fk + alpha * c1_gradfk_pk;

% Preallocations
xseq        = zeros(length(x0), kmax+1);
btseq       = zeros(1, kmax);
alphas      = zeros(1, kmax);
pks         = zeros(length(x0), kmax);
inner_iters = zeros(1, kmax); 


k = 0;
xseq(:,1) = xk;   

while k < kmax && gradfk_norm >= tolgrad

    eta_k = min(0.5, sqrt(gradfk_norm))*gradfk_norm;

    % The system we need to solve is Hess(fk)*pk = -graf(fk) <-> Hk*z=ck
    % Hv(p) returns H(xk)*p via a centered finite difference (matrix-free).

    Hv = @(p) hess_vec_fd(xk, p, gradf, h_fd);
    z = zeros(length(x0),1); 
    rk =gradfk;
    dk = -rk;

    r_old = rk'*rk;

    % Boolean variable used to understand whether the inner loop got to convergence,
    % it has to go back to false at each iteration k.
    stop_inner = false; 
    
    j = 0;
    while ~stop_inner && j < max_cg % Inner loop for solving the system with CG.

        Hdk = Hv(dk);
        curv = dk'*Hdk;
        
	% If the curvature is non-positive (or near zero), the CG loop is terminated: fall back to steepest descent at the first iteration, or accept the last valid approximation otherwise.   
     	if curv <= 1e-10 * norm(dk)^2    
            if j == 0
                p_tn = -gradfk;                
            else
                p_tn = z;
            end
            break

        else
            alpha_j = r_old/curv;
            z = z + alpha_j * dk;
            rk = rk + alpha_j * Hdk;

            % Check convergence
            if norm(rk) <= eta_k 
                p_tn = z;       
                %stop_inner = true;
                break;
            end
            
            % Updates for the next iteration.
            rk_new = rk'*rk;
            beta_j = rk_new/r_old;
            dk = -rk + beta_j * dk;
            r_old = rk_new; 
            
            j = j+1;

       end

    end

    j_cg = j;

    if j == max_cg
        disp('MAX CG (Truncated Newton - Hess free): maximum number of cg iterations reached.');
        p_tn = z;   % CG done without meeting the curvature or tolerance condition: use the last valid iterate
    end
    % Assign the computed step/search direction
    pk = p_tn;

    % Check if the search direction is null
    if norm(pk) <= 1e-12
        disp('Truncated Newton Hess free: null direction, stop.');
        if gradfk_norm < tolgrad
             return;
        else
            warning('Truncated Newton: pk near zero but gradient not converged.');
        end

    end


    % BACKTRACKING.

    alpha = 1; % Reset the value of alpha (parameter used for linesearch)
    xnew = xk + alpha * pk;   % Computing the candidate solution at step k.
    fnew = f(xnew);
    c1_gradfk_pk = c1 * gradfk' * pk;

    bt = 0;
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha; % Step reduction
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end

    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        disp('Backtracking (Truncated Newton): maximum number of iterations reached.');
        k = k+1;
        xseq(:, k+1) = xk;
        btseq(k) = bt;
        pks(:, k) = pk;
        alphas(k) = alpha;
        inner_iters(k) = j_cg;

        break;            
    end
            
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    k = k + 1;
    
    % Storing iteration data
    xseq(:, k+1)   = xk;
    btseq(k)       = bt;
    pks(:, k)      = pk;
    alphas(k)      = alpha;
    inner_iters(k) = j_cg;
        

end %while loop on k

% Trimming the final structures.
xseq   = xseq(:, 1:k+1);
btseq  = btseq(1:k);
pks    = pks(:, 1:k);
inner_iters = inner_iters(:,1:k);
end %function end






function Hp = hess_vec_fd(xk, p, gradf, h_fd)
% HESS_VEC_FD  Approximates the Hessian-vector product H(xk)*p using a
% centered finite difference on the gradient, without ever building H
% explicitly:
%
%   Hp = ( gradf(xk + h*p) - gradf(xk - h*p) ) / (2*h)

normp = norm(p);

if normp < eps
    Hp = zeros(size(p));
    return;
end

h = h_fd / normp;
Hp = (gradf(xk + h*p) - gradf(xk - h*p)) / (2*h);
end