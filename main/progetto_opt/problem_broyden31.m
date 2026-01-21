function [f,gradf,hessf,xbar] = problem_broyden31()
% PROBLEM_BROYDEN31
% ------------------------------------------------------------
% Problem 31: Broyden tridiagonal least squares
%
% r_k(x) = (3-2x_k)x_k - x_{k-1} - 2x_{k+1} + 1, con x_0 = x_{n+1} = 0
% F(x) = 1/2 * ||r(x)||^2
%
% grad: J^T r
% Hess: J^T J - 4*diag(r) (può essere indefinita -> tau-correction utile)
%
% OUTPUT:
%   f, gradf, hessf : handle
%   xbar            : funzione handle che dato n ritorna xbar (qui -ones)

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;

% xbar richiesto: x_l = -1
xbar = @(n) -ones(n,1);

    function r = rvec(x)
        n = length(x);
        xm1 = [0; x(1:n-1)];
        xp1 = [x(2:n); 0];
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
