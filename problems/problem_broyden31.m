function [f,gradf,hessf,xbar,rfun,xstarfun] = problem_broyden31()
% ------------------------------------------------------------
% Problem 31: Broyden tridiagonal least-squares problem
%
% OUTPUT:
%   f        : F(x) = 0.5*||r(x)||^2
%   gradf    : gradiente di F
%   hessf    : Hessiana di F
%   xbar     : punto iniziale
%   rfun     : vettore dei residui r(x)
%   xstarfun : handle xstarfun(n) -> soluzione globale x*
%
% La soluzione x* viene calcolata con LSQNONLIN risolvendo
%
%       min_x 0.5*||r(x)||^2
%
% fornendo esplicitamente il Jacobiano sparso del residuo.
%
% Poiche' F(x) >= 0 per ogni x, se ||r(x*)|| ~= 0 allora
%
%       F(x*) ~= 0
%
% e quindi x* e' un minimo globale.
%
% Il risultato viene cachato: chiamate ripetute con lo stesso n
% non rieseguono lsqnonlin.
% ------------------------------------------------------------

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;
rfun  = @rvec;

xbar = @(n) -ones(n,1);
xstarfun = @xstar_cached;


    % --------------------------------------------------------
    % Residuo
    % --------------------------------------------------------
    function r = rvec(x)

        n = length(x);

        xm1 = [0; x(1:n-1)];
        xp1 = [x(2:n); 0];

        r = (3 - 2*x).*x - xm1 - 2*xp1 + 1;

    end


    % --------------------------------------------------------
    % Funzione obiettivo
    % --------------------------------------------------------
    function Fx = Ffun(x)

        r = rvec(x);

        Fx = 0.5 * (r' * r);

    end


    % --------------------------------------------------------
    % Gradiente
    % --------------------------------------------------------
    function g = gfun(x)

        n = length(x);
        r = rvec(x);

        d = 3 - 4*x;

        J = spdiags( ...
            [-ones(n,1), d, -2*ones(n,1)], ...
            [-1,0,1], n, n);

        g = J' * r;

    end


    % --------------------------------------------------------
    % Hessiana esatta
    % --------------------------------------------------------
    function H = Hfun(x)

        n = length(x);
        r = rvec(x);

        d = 3 - 4*x;

        J = spdiags( ...
            [-ones(n,1), d, -2*ones(n,1)], ...
            [-1,0,1], n, n);

        H = J' * J - 4 * spdiags(r,0,n,n);

    end


    % --------------------------------------------------------
    % Soluzione globale tramite LSQNONLIN
    % --------------------------------------------------------
    function xs = xstar_cached(n)

        persistent cache_n cache_x

        % ----------------------------------------------------
        % Cache
        % ----------------------------------------------------
        if ~isempty(cache_n) && cache_n == n
            xs = cache_x;
            return;
        end

        fprintf('\n');
        fprintf('===============================================\n');
        fprintf(' Broyden31: ricerca soluzione globale\n');
        fprintf(' n = %d\n', n);
        fprintf('===============================================\n');

        % ----------------------------------------------------
        % Punto iniziale
        % ----------------------------------------------------
        x0 = xbar(n);

        % ----------------------------------------------------
        % Opzioni LSQNONLIN
        %
        % trust-region-reflective e' adatto al problema grande
        % e sfrutta il Jacobiano sparso fornito da residual_J.
        % ----------------------------------------------------
        opts = optimoptions('lsqnonlin', ...
            'Algorithm', 'trust-region-reflective', ...
            'SpecifyObjectiveGradient', true, ...
            'Display', 'iter', ...
            'FunctionTolerance', 1e-14, ...
            'StepTolerance', 1e-14, ...
            'OptimalityTolerance', 1e-12, ...
            'MaxIterations', 2000, ...
            'MaxFunctionEvaluations', 1e6);

        % ----------------------------------------------------
        % Risoluzione nonlinear least squares
        % ----------------------------------------------------
        [xs,resnorm,residual,exitflag,output] = ...
            lsqnonlin(@residual_J,x0,[],[],opts);

        % ----------------------------------------------------
        % Verifica della soluzione
        % ----------------------------------------------------
        Gstar = 0.5 * resnorm;

        rinf = norm(residual,inf);
        r2   = norm(residual,2);

        fprintf('\n');
        fprintf('===============================================\n');
        fprintf(' Risultato\n');
        fprintf('===============================================\n');
        fprintf('exitflag        = %d\n', exitflag);
        fprintf('iterations      = %d\n', output.iterations);
        fprintf('||r(x*)||_inf   = %.6e\n', rinf);
        fprintf('||r(x*)||_2     = %.6e\n', r2);
        fprintf('G(x*)           = %.6e\n', Gstar);
        fprintf('===============================================\n');

        % ----------------------------------------------------
        % Controllo del minimo globale
        %
        % G(x) = 0.5*||r(x)||^2 >= 0 per ogni x.
        %
        % Se il residuo e' numericamente nullo, allora
        % G(x*) = 0 ed x* e' un minimo globale.
        % ----------------------------------------------------
        if rinf < 1e-10

            fprintf('\n');
            fprintf('SOLUZIONE DI RIFERIMENTO TROVATA.\n');
            fprintf('Il residuo e'' numericamente nullo.\n');
            fprintf('Pertanto G(x*) = 0 entro la precisione numerica.\n');
            fprintf('x* e'' un minimo globale.\n');

        else

            warning(['LSQNONLIN non ha trovato un residuo ' ...
                     'sufficientemente piccolo. ' ...
                     'La globalita'' non e'' stata certificata.']);

        end

        % ----------------------------------------------------
        % Cache
        % ----------------------------------------------------
        cache_n = n;
        cache_x = xs;

    end


    % --------------------------------------------------------
    % Residuo + Jacobiano
    %
    % Questa funzione viene passata direttamente a LSQNONLIN.
    % Il Jacobiano e' tridiagonale e sparso.
    % --------------------------------------------------------
    function [r,J] = residual_J(x)

        n = length(x);

        % Residuo
        xm1 = [0; x(1:n-1)];
        xp1 = [x(2:n); 0];

        r = (3 - 2*x).*x - xm1 - 2*xp1 + 1;

        % Jacobiano
        d = 3 - 4*x;

        J = spdiags( ...
            [-ones(n,1), d, -2*ones(n,1)], ...
            [-1,0,1], n, n);

    end

end