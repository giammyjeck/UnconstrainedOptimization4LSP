clear; clc; close all;

% ============================================================
% Debug iterazione-per-iterazione: truncated_newton_method su
% trig16, case2 (gradiente e Hessiana entrambi approssimati per
% differenze finite), per capire perche' il tuning non trova
% configurazioni ammissibili.
%
% A differenza del test sui problemi giocattolo, qui confrontiamo
% ad ogni iterazione:
%   - ||grad_fd||    : quello che l'algoritmo vede e su cui decide
%   - ||grad_exact|| : quello vero, per capire se grad_fd sta
%                      "mentendo" all'algoritmo (rumore che gli
%                      impedisce di accorgersi di essere vicino
%                      alla convergenza, o viceversa)
% ============================================================

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbar_gen] = problem_trig16(); %#ok<ASGLU>

n = 1e5;

% k e type gia' individuati come ottimali per case2 (schema centrato)
k_fd    = 4;
type_fd = 1;   % 1 = h costante, 2 = hi relativo

gradf = @(x) trig_fd_case2_grad_only(x, k_fd, type_fd);
hessf = @(x) trig_fd_case2_hess_only(x, k_fd, type_fd);

% Starting point: uno dei punti "tipici" usati negli esperimenti
rng(346710);
xb = xbar_gen(n);
x0 = xb + (2*rand(n,1) - 1);

% Parametri generosi apposta, per vedere il comportamento completo
% senza che saturi troppo presto e nasconda cosa succede davvero
tolgrad = 1e-6;
kmax    = 500;
c1      = 1e-4;
rho     = 0.8;
btmax   = 50;
max_cg  = 1000;

fprintf('\n===============================================\n');
fprintf(' Debug truncated_newton_method - trig16 case2\n');
fprintf(' n=%d, k_fd=%d, type_fd=%d\n', n, k_fd, type_fd);
fprintf('===============================================\n\n');

[xk, fk, gn_fd, k, xseq, btseq, pks, inner_iters] = truncated_newton_method( ...
    x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, max_cg);

fprintf('%-4s %-14s %-12s %-12s %-6s %-4s %-14s\n', ...
    'k', 'f(xk)', '||g_fd||', '||g_exact||', 'cg_it', 'bt', 'grad_fd''*pk');

for i = 1:k
    x_before = xseq(:, i);
    x_after  = xseq(:, i+1);

    g_fd_before = gradf(x_before);
    pk          = pks(:, i);

    descent_check = g_fd_before' * pk;

    gnorm_fd_after    = norm(gradf(x_after));
    gnorm_exact_after = norm(grad_exact(x_after));

    fprintf('%-4d %-14.6e %-12.3e %-12.3e %-6d %-4d %-14.3e\n', ...
        i, f(x_after), gnorm_fd_after, gnorm_exact_after, ...
        inner_iters(i), btseq(i), descent_check);
end

fprintf('\nRisultato finale:\n');
fprintf('  k = %d (kmax = %d)\n', k, kmax);
fprintf('  ||grad_fd||    = %.3e  (tolgrad = %.0e)\n', gn_fd, tolgrad);
fprintf('  ||grad_exact|| = %.3e\n', norm(grad_exact(xk)));
fprintf('  f(xk)          = %.6e\n', fk);
fprintf('  bt saturato in almeno una iterazione : %d\n', any(btseq >= btmax));
fprintf('  cg saturato in almeno una iterazione : %d\n', any(inner_iters >= max_cg));
