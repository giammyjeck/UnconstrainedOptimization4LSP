function [f,gradf,hessf,xbar] = problem_trig16()
% PROBLEM_TRIG16
% ------------------------------------------------------------
% Problem 16: Banded Trigonometric
%
% F(x) = sum_{i=1}^n i(1 - cos(x_i)) + 2*sum_{i=1}^{n-1} sin(x_i) - (n-1)*sin(x_n)
% grad e Hess sono esatti; Hess è diagonale (sparsa).
%
% OUTPUT:
%   f, gradf, hessf : handle
%   xbar            : funzione handle che dato n ritorna xbar (qui ones)

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;

% xbar richiesto: xi = 1
xbar = @(n) ones(n,1);

    function Fx = Ffun(x)
        n = length(x);
        i = (1:n)';
        Fx = sum( i .* (1 - cos(x)) ) ...
           + 2 * sum( sin(x(1:n-1)) ) ...
           - (n-1) * sin(x(n));
    end

    function g = gfun(x)
        n = length(x);
        i = (1:n)';

        g = zeros(n,1);
        g(1:n-1) = i(1:n-1).*sin(x(1:n-1)) + 2*cos(x(1:n-1));
        g(n)     = n*sin(x(n)) - (n-1)*cos(x(n));
    end

    function H = Hfun(x)
        n = length(x);
        i = (1:n)';

        d = zeros(n,1);
        d(1:n-1) = i(1:n-1).*cos(x(1:n-1)) - 2*sin(x(1:n-1));
        d(n)     = n*cos(x(n)) + (n-1)*sin(x(n));

        H = spdiags(d, 0, n, n);
    end

end
