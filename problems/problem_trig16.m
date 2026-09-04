function [f,gradf,hessf,xbar,xstarfun] = problem_trig16()
% ------------------------------------------------------------
% Problem 16: Banded Trigonometric
%
% F(x) = sum_{i=1}^n i * (1 - cos(x_i))
%      + 2 * sum_{i=1}^{n-1} sin(x_i)
%      - (n - 1) * sin(x_n)
%
% OUTPUT:
%   f        : F(x)
%   gradf    : gradiente di F
%   hessf    : Hessiana di F (diagonale)
%   xbar     : punto iniziale xbar(n)
%   xstarfun : handle xstarfun(n) -> minimo locale x* piu' vicino a xbar
%
% A differenza di Broyden31, questo problema NON e' un least-squares
% esplicito (non esiste un residuo r(x) tale che F = 0.5*||r||^2 e
% coerente con Ffun/gfun), quindi non possiamo usare LSQNONLIN come
% fatto per Broyden31.
%
% F e' inoltre periodica in ogni componente (somma di seni e coseni),
% quindi ha moltissimi punti stazionari: minimi, selle e massimi.
% Risolvere semplicemente grad(x)=0 (es. con FSOLVE) non basta, perche'
% si puo' convergere a una sella invece che a un minimo, a seconda del
% punto di partenza.
%
% x* viene quindi calcolato con FMINUNC (algoritmo trust-region,
% gradiente e Hessiana esatti forniti), che minimizza F davvero: in
% presenza di curvatura negativa si sposta nella direzione che fa
% scendere F, quindi converge al minimo locale raggiungibile dal punto
% di partenza, non a una sella. Il punto di partenza usato e' xbar(n),
% lo stesso da cui partono gli esperimenti, cosi' x* rappresenta il
% minimo locale "di riferimento" per quel punto di partenza.
%
% Dopo la risoluzione verifichiamo comunque che l'Hessiana in x* sia
% definita positiva. Essendo l'Hessiana diagonale, questo equivale a
% controllare che tutti gli elementi della diagonale siano positivi:
% condizione necessaria e sufficiente, non un'euristica.
%
% NOTA: x* e' un minimo LOCALE, non c'e' garanzia di globalita' (a
% differenza di Broyden31, dove F>=0 sempre da' quella garanzia).
%
% Il risultato viene cachato: chiamate ripetute con lo stesso n non
% rieseguono fminunc.
% ------------------------------------------------------------
 
f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;
 
xbar = @(n) ones(n,1);
xstarfun = @xstar_cached;
 
 
    % --------------------------------------------------------
    % Funzione obiettivo
    % --------------------------------------------------------
    function Fx = Ffun(x)
        n = length(x);
        i = (1:n)';
        Fx = sum( i .* (1 - cos(x)) ) ...
           + 2 * sum( sin(x(1:n-1)) ) ...
           - (n-1) * sin(x(n));
    end
 
 
    % --------------------------------------------------------
    % Gradiente
    % --------------------------------------------------------
    function g = gfun(x)
        n = length(x);
        i = (1:n)';
        g = zeros(n,1);
        g(1:n-1) = i(1:n-1).*sin(x(1:n-1)) + 2*cos(x(1:n-1));
        g(n)     = n*sin(x(n)) - (n-1)*cos(x(n));
    end
 
 
    % --------------------------------------------------------
    % Hessiana esatta (diagonale)
    % --------------------------------------------------------
    function H = Hfun(x)
        n = length(x);
        i = (1:n)';
        d = zeros(n,1);
        d(1:n-1) = i(1:n-1).*cos(x(1:n-1)) - 2*sin(x(1:n-1));
        d(n)     = n*cos(x(n)) + (n-1)*sin(x(n));
        H = spdiags(d, 0, n, n);
    end
 
 
    % --------------------------------------------------------
    % Punto stazionario tramite FSOLVE su grad F(x) = 0
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
        fprintf(' Trig16: ricerca minimo locale piu'' vicino a xbar\n');
        fprintf(' n = %d\n', n);
        fprintf('===============================================\n');
 
        % ----------------------------------------------------
        % Punto di partenza: lo stesso usato per gli esperimenti.
        % ----------------------------------------------------
        x0 = xbar(n);
 
        % ----------------------------------------------------
        % FMINUNC, non FSOLVE.
        %
        % FSOLVE risolve grad(x)=0 e puo' fermarsi su un punto
        % qualsiasi (minimo, sella, massimo). FMINUNC invece minimizza
        % F davvero: con l'algoritmo trust-region e l'Hessiana esatta
        % fornita, in presenza di curvatura negativa si muove nella
        % direzione che fa scendere F, quindi converge a un minimo
        % locale (quello raggiungibile da x0), non a una sella.
        % ----------------------------------------------------
        opts = optimoptions('fminunc', ...
            'Algorithm', 'trust-region', ...
            'SpecifyObjectiveGradient', true, ...
            'HessianFcn', 'objective', ...
            'Display', 'iter', ...
            'FunctionTolerance', 1e-14, ...
            'StepTolerance', 1e-14, ...
            'OptimalityTolerance', 1e-12, ...
            'MaxIterations', 2000, ...
            'MaxFunctionEvaluations', 1e6);
 
        [xs, Fval, exitflag, output] = fminunc(@F_grad_hess, x0, opts);
 
        % ----------------------------------------------------
        % Verifica: e' davvero un minimo?
        %
        % Hessiana diagonale -> definita positiva se e solo se
        % tutti gli elementi della diagonale sono positivi.
        % ----------------------------------------------------
        diagHs = full(diag(Hfun(xs)));
        is_min = all(diagHs > 0);
        gnorm  = norm(gfun(xs), inf);
 
        fprintf('\n');
        fprintf('===============================================\n');
        fprintf(' Risultato\n');
        fprintf('===============================================\n');
        fprintf('exitflag        = %d\n', exitflag);
        fprintf('iterations      = %d\n', output.iterations);
        fprintf('||grad(x*)||_inf = %.6e\n', gnorm);
        fprintf('F(x*)            = %.6e\n', Fval);
        fprintf('Hessiana(x*) definita positiva = %d\n', is_min);
        fprintf('===============================================\n');
 
        if gnorm >= 1e-8
            warning(['FMINUNC non ha trovato un punto stazionario ' ...
                     'sufficientemente accurato.']);
        end
 
        if ~is_min
            warning(['Il punto trovato NON risulta un minimo locale ' ...
                     '(Hessiana non definita positiva): risultato ' ...
                     'inatteso per FMINUNC, da controllare.']);
        end
 
        % ----------------------------------------------------
        % Cache
        % ----------------------------------------------------
        cache_n = n;
        cache_x = xs;
 
    end
 
 
    % --------------------------------------------------------
    % F(x), grad(x) e Hessiana, nel formato richiesto da FMINUNC
    % --------------------------------------------------------
    function [Fx, g, H] = F_grad_hess(x)
        Fx = Ffun(x);
        g  = gfun(x);
        H  = Hfun(x);
    end
 
 
end
 