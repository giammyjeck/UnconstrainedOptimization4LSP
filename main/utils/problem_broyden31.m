function [f,gradf,hessf,xbar] = problem_broyden31()
% ------------------------------------------------------------
% Problem 31: Broyden tridiagonal least–squares problem
%
% The residuals are defined as:
%   r_k(x) = (3 - 2*x_k)*x_k - x_{k-1} - 2*x_{k+1} + 1,
%   with boundary conditions: x_0 = x_{n+1} = 0.
%
% The objective function has least–squares form:
%   F(x) = 0.5 * ||r(x)||^2 = 0.5 * sum_{k=1}^n r_k(x)^2
%
% Gradient and Hessian:
%   grad F(x) = J(x)' * r(x)
%   Hess F(x) = J(x)' * J(x) - 4 * diag(r(x))
%
% OUTPUT:
%   f, gradf, hessf : handle to F(x), grad(F(x)), hess(F(x))
%   xbar   : handle that returns the standard starting point xbar(n)

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;

% Required starting point:
%   xbar_k = -1  for all k
xbar = @(n) -ones(n,1);

    % Residual vector
    function r = rvec(x)
        n = length(x);
        xm1 = [0; x(1:n-1)]; %x_{k-1}
        xp1 = [x(2:n); 0]; %x_{k+1}
        r = (3 - 2*x).*x - xm1 - 2*xp1 + 1;
    end
    
    function Fx = Ffun(x)
        r = rvec(x);
        Fx = 0.5 * (r' * r);
    end
    
    function g = gfun(x)
        n = length(x);
        r = rvec(x);
        d = 3 - 4*x;
        J = spdiags([-ones(n,1), d, -2*ones(n,1)], [-1,0,1], n, n);
        g = J' * r;
    end

    function H = Hfun(x)
        n = length(x);
        r = rvec(x);
        d = 3 - 4*x;
        J = spdiags([-ones(n,1), d, -2*ones(n,1)], [-1,0,1], n, n);
        H = J' * J - 4 * spdiags(r, 0, n, n);
    end

end
