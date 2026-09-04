%% DIAGNOSTICA: loop di correzione dell'Hessiana (Cholesky + tau*I)
%
% Isola ESATTAMENTE la stessa logica presente dentro
% modified_newton_method.m (stessa formula di tau_0, stesso raddoppio,
% stesso maxit=200), ma la applica a UN SOLO punto per volta, cosi' puoi
% vedere subito:
%   - quante iterazioni j di raddoppio servono prima che chol() riesca
%   - se maxit=200 (hardcoded nella funzione) e' davvero un collo di
%     bottiglia con beta=1e-3 alla scala di n=1e5
%   - quanto costa in tempo la Cholesky sparsa stessa (per capire se il
%     problema e' il NUMERO di tentativi o il COSTO di ogni tentativo)
%
% Non fa girare modified_newton_method per intero: lavora direttamente
% su hessf(x) in uno o due punti scelti a mano, quindi e' quasi
% immediato anche a n=1e5.

clear; clc;

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbarfun, xstarfun] = problem_trig16();

n = 1e5;
seed = 346710;
rng(seed);

xb = xbarfun(n);
X0 = [xb, xb + (2*rand(n,5) - 1)];

% --- Punti da ispezionare ---
% x0: punto di partenza standard
% x1: un piccolo passo di discesa da x0, per vedere se la scala della
%     diagonale cambia significativamente durante le prime iterazioni
%     (utile per capire se il problema di scala e' solo iniziale o persiste)
x0 = X0(:,1);
g0 = grad_exact(x0);
x1 = X0(:,5);

points = struct('label', {'x0 (start)', 'x1 (piccolo passo da x0)'}, ...
                 'x',     {x0, x1});

% --- Valori di beta da confrontare ---
beta_list = [1e-2, 1, ];

maxit_internal = 200;   % stesso valore hardcoded in modified_newton_method.m

for p = 1:numel(points)
    x = points(p).x;
    fprintf('\n========================================\n');
    fprintf('Punto: %s\n', points(p).label);
    fprintf('========================================\n');

    tic;
    Hk = hess_exact(x);
    t_hess = toc;
    fprintf('hess_exact(x): %.3fs, sparse=%d, nnz=%d (%.4f%% di %d^2)\n', ...
        t_hess, issparse(Hk), nnz(Hk), 100*nnz(Hk)/n^2, n);

    diagHk = full(diag(Hk));
    fprintf('diag(Hk): min=%.3e, max=%.3e, min(abs)=%.3e, max(abs)=%.3e\n', ...
        min(diagHk), max(diagHk), min(abs(diagHk)), max(abs(diagHk)));

    isPositive = all(diagHk > 1e-6);
    fprintf('Diagonale tutta positiva (soglia tol=1e-6)? %d\n', isPositive);

    for beta = beta_list
        if isPositive
            tau0 = 0;
        else
            tau0 = beta - min(diagHk);
        end

        tau = tau0;
        j_used = NaN;
        t_chol_total = 0;

        for j = 1:maxit_internal
            if n >= 1e4
                Bk = Hk + tau*speye(n);
            else
                Bk = Hk + tau*eye(n);
            end

            tic;
            [R, flag] = chol(Bk); %#ok<ASGLU>
            t_chol_total = t_chol_total + toc;

            if flag == 0
                j_used = j;
                break;
            else
                tau = max(beta, 2*tau);
            end
        end

        if isnan(j_used)
            fprintf('  beta=%-8.1e | NON risolto entro maxit=%d | tau finale=%.3e | t_chol_tot=%.2fs\n', ...
                beta, maxit_internal, tau, t_chol_total);
        else
            fprintf('  beta=%-8.1e | j=%3d tentativi | tau finale=%.3e | t_chol_tot=%.2fs (%.4fs/tentativo)\n', ...
                beta, j_used, tau, t_chol_total, t_chol_total/j_used);
        end
    end
end

fprintf('\n--- Come leggere il risultato ---\n');
fprintf('- Se "j" e'' sempre 1: la matrice era gia'' positiva definita (o quasi),\n');
fprintf('  il beta piccolo non e'' un problema li''.\n');
fprintf('- Se "j" cresce molto per beta piccoli e cala per beta grandi:\n');
fprintf('  confermato, beta=1e-3 e'' troppo piccolo rispetto alla scala reale\n');
fprintf('  della matrice a questo n, e i raddoppi mangiano iterazioni di\n');
fprintf('  maxit=200 prima di arrivare a un valore utile. Alzare beta ha\n');
fprintf('  senso.\n');
fprintf('- Se "NON risolto entro maxit" compare anche per beta grandi:\n');
fprintf('  il problema non e'' beta, ma maxit=200 troppo basso in assoluto\n');
fprintf('  (andrebbe reso parametro esposto, non hardcoded).\n');
fprintf('- t_chol_tot/tentativo alto (es. > 0.1-1s) a prescindere da beta:\n');
fprintf('  il collo di bottiglia e'' il COSTO della Cholesky sparsa stessa\n');
fprintf('  (fill-in, manca un riordinamento), non il numero di tentativi -\n');
fprintf('  in quel caso ne parliamo separatamente (symamd/amd prima di chol).\n');