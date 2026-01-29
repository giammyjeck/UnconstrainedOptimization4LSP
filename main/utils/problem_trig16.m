function [f,gradf,hessf,xbar, tfun] = problem_trig16()
% ------------------------------------------------------------
% Problem 16: Banded Trigonometric
%
% The objective function is:
%   F(x) = sum_{i=1}^n i * (1 - cos(x_i))
%        + 2 * sum_{i=1}^{n-1} sin(x_i)
%        - (n - 1) * sin(x_n)
%
% OUTPUT:
%   f, gradf, hessf : handle to F(x), grad(F(x)), hess(F(x))
%   xbar   : handle that returns the standard starting point xbar(n)

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;
tfun  = @rvec;

xbar = @(n) ones(n,1);

    function r = rvec(x)
            % r_i(x) = i[(1-cos x_i) + sin x_{i-1} - sin x_{i+1}], con x0=x_{n+1}=0
            x = x(:);
            n = length(x);
            i = (1:n)';
    
            xm1 = [0; x(1:n-1)];
            xp1 = [x(2:n); 0];
    
            r = i .* ((1 - cos(x)) + sin(xm1) - sin(xp1));
    end

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