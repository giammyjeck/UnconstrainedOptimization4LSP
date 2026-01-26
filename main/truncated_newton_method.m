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

    % Function handle for the armijo condition
    farmijo = @(fk, alpha, c1_gradfk_pk) ...
        fk + alpha * c1_gradfk_pk;

    % Variables iniziatization
    xseq = zeros(length(x0), kmax);
    btseq = zeros(1, kmax);
    alphas = zeros(1, kmax);
    pks = zeros(length(x0), kmax);
    inner_iters = zeros(1, kmax); %vettore che tiene conto delle iterazioni interne di ogni interazione esterna k
    inner_it = 0; %indice che tiene conto delle iterazioni
    
    xk = x0;
    fk = f(xk);
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    k = 1;
    while k <= kmax && gradfk_norm >= tolgrad


        % The system we need to solve is Hess(fk)*pk = -graf(fk) <-> Bk*z=ck
        z = zeros(length(x0),1); 
        Bk = hessf(xk);
        ck = -gradfk;

        % Initialize p_tn with the steepest descent direction (-gradfk) 
        % as a fallback to guarantee a valid descent direction even if 
        % the CG inner loop fails or terminates at the first iteration.
        p_tn = ck; 

        eta_k = min(0.5, sqrt(gradfk_norm));

        rk = ck - Bk * z; % Residual of the system, at the first iteration z = 0 so rk = ck.
        rk_old = rk'*rk;
        dk = rk; % d is the conjugate direction, at the first iteration z = 0 so dk = ck.

        stop_inner = false; % Boolean variable used to understand whether the inner loop got to convergence, it has to go back to false at each iteration k
        
        j = 0;
        while ~stop_inner && j < max_cg % Inner loop for solving the system with CG

            j = j+1;
            Bdk = Bk*dk;
            curv = dk'*Bdk;
            
            if curv > 0 % If the curvature is positive we can proceed with CG method.
                alpha_j = rk_old/curv;
                z = z + alpha_j * dk;
                rk = rk - alpha_j * Bdk; % chat dice che va il - ma la pieraccini ha scritto +

                % Check on convergence
                if norm(rk) <= eta_k * norm(ck)
                    p_tn = z;      
                    stop_inner = true;
                    break;
                end
                
                % Update for the next iteration
                rk_new = rk'*rk;
                beta_j = rk_new/rk_old;
                dk = rk + beta_j * dk;
                rk_old = rk_new; 
                p_tn = z;

            else % Negative or zero curvature case
                if j == 1 
                    p_tn = -gradfk;
                    
                else
                    p_tn = z;
                end            
                stop_inner = true;

                break
            end
        end

        inner_iters(k) = j;
        pk = p_tn;
        
        % Backtracking strategy
        if norm(pk) == 0
            disp('Truncated Newton: null direction, stop.');
            return;
        end

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

% Truncate sequences to the actual number of iterations
xseq   = xseq(:, 1:k);
btseq  = btseq(1:k);
alphas = alphas(1:k);
pks    = pks(:, 1:k);

xseq = [x0, xseq]; 

end %function end