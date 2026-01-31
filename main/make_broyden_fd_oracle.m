
%  This function generates optimized function handles for the gradient and 
%  the Hessian using Finite Differences and the Gauss-Newton approximation.


function [gradf, hessf] = make_broyden_fd_oracle(rfun, k, mode, bwJ)
    % Returns two function handles for the Gradient and the Hessian:
    %   - gradf(x) = J(x)' * r(x)
    %   - hessf(x) = J(x)' * J(x)  (Gauss-Newton Approximation)
    
    % IMPLEMENTATION NOTE: A caching system is used for x. 
    % Since the Modified Newton method typically evaluates both gradf(xk) 
    % and hessf(xk) at the same point during a single iteration, 
    % the cache prevents re-calculating the expensive Jacobian if the point
    % x hasn't changed.
    
    % Initialization of the cache variables.
    last_x = []; % last_x stores the coordinates of the most recent evaluation.
    last_g = []; % last_g: stores the gradient computed at last_x.
    last_H = []; % last_H: stores the Hessian computed at last_x.
    
        function [g,H] = compute_at(x)

            x = x(:);

    
            % Cache hit: if x is identical to the last evaluation, return stored values
            if ~isempty(last_x) && isequal(size(last_x),size(x)) && all(x == last_x)
                g = last_g; H = last_H;
                return
            end
            
            % Cache miss: compute numerical Gradient and Hessian.
            [g,H] = broyden_fd_grad_hess_GN(rfun, x, k, mode, bwJ);
            
            % Update cache with new results.
            last_x = x;
            last_g = g;
            last_H = H;
            end

    % Define handles for gradient and hessian.
    gradf = @(x) get_g(x);
    hessf = @(x) get_H(x);
    
        function g = get_g(x)
            [g,~] = compute_at(x);
        end
    
        function H = get_H(x)
            [~,H] = compute_at(x);
        end
    end
    
    function [g,H] = broyden_fd_grad_hess_GN(rfun, x, k, mode, bwJ)
    % Estimating the Jacobian (J) of the residual r(x) using Central Finite Differences.
    %
    % 1. Sparsity: bwJ = 1 for Broyden31 (Tridiagonal structure).
    % 2. Efficiency: 1D Coloring with stride = 2*bwJ + 1 (only 3 evaluations of r).
    % 3. Computing:
    %       Gradient: g = J' * r;
    %       Hessian:  H = J' * J.
    
    
        x = x(:);
        n = length(x);

        % Calculate finite difference steps h.
        h = steps_fd(x,k,mode);
        r0 = rfun(x);
        
        % Coloring Strategy to exploit sparsity.
        stride = 2*bwJ + 1;
        J = sparse(n,n);
        
        for c = 1:stride
            idx = c:stride:n;
        
            dx = zeros(n,1);
            dx(idx) = h(idx);
            
            % Finite differences for the selected group.
            rp = rfun(x + dx);
            rm = rfun(x - dx);
            dr = rp - rm;
            
            % Unpack results into the sparse Jacobian matrix.
            for jj = idx
                rows = max(1,jj-bwJ):min(n,jj+bwJ);
                J(rows,jj) = dr(rows) / (2*h(jj));
            end
        end
        
        % Gradient: J' * r (Steepest descent direction)
        g = J' * r0;
        
        % Hessian Approximation: Gauss-Newton (J' * J)
        % This is Symmetric Positive Semi-Definite (SPSD) by construction.
        H = J' * J;
        H = 0.5*(H + H'); % Enforce exact symmetry against numerical noise. 
end
    
function show_cache_note()

end