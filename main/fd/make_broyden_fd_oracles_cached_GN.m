%% ========================================================================
%  CASE 2 - BROYDEN31: J FD su residuo + Gauss-Newton, con CACHE
% ========================================================================
function [gradf, hessf] = make_broyden_fd_oracles_cached_GN(rfun, k, mode, bwJ)
% Ritorna due function-handle:
%   gradf(x) = J(x)' r(x)
%   hessf(x) = J(x)' J(x)    (Gauss-Newton)
%
% E usa CACHE su x: Modified Newton tipicamente fa:
%   g = gradf(xk) ; H = hessf(xk)
% quindi senza cache stimerei due volte J (carissimo).

last_x = [];
last_g = [];
last_H = [];

    function [g,H] = compute_at(x)
        x = x(:);

        % cache hit: stesso x identico (tipico nella stessa iterazione)
        if ~isempty(last_x) && isequal(size(last_x),size(x)) && all(x==last_x)
            g = last_g; H = last_H;
            return
        end

        [g,H] = broyden_fd_grad_hess_GN(rfun, x, k, mode, bwJ);

        last_x = x;
        last_g = g;
        last_H = H;
    end

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
% Stima Jacobiana J del residuo r(x) con FD centrali bandate.
% Banda: bwJ = 1 per Broyden31 (tridiagonale).
% Coloring 1D: stride = 2*bwJ+1 -> qui stride=3, quindi ~3 valutazioni di r.
%
% Poi costruisco:
%   g = J' r
%   H = J'J  (Gauss-Newton, PSD)

x = x(:);
n = length(x);

h = steps_fd(x,k,mode);
r0 = rfun(x);

stride = 2*bwJ + 1;
J = sparse(n,n);

for c = 1:stride
    idx = c:stride:n;

    dx = zeros(n,1);
    dx(idx) = h(idx);

    rp = rfun(x + dx);
    rm = rfun(x - dx);
    dr = rp - rm;

    for jj = idx
        rows = max(1,jj-bwJ):min(n,jj+bwJ);
        J(rows,jj) = dr(rows) / (2*h(jj));
    end
end

g = J' * r0;
H = J' * J;
H = 0.5*(H + H'); % simmetrizza per eliminare asimmetria numerica
end

function show_cache_note()
% helper solo per rendere esplicito a chi legge cosa succede.
% non cambia nulla a runtime
end