% trace_truncated_n2_start2.m
%
% Isola lo starting point 2 (n=2, exact) usando lo STESSO seed e la
% STESSA generazione di X0 di run_experiments_31.m, cosi' il punto
% analizzato qui e' identico a quello usato nell'esperimento principale.
%
% Esegue Truncated Newton in versione DIAGNOSTICA (logga ogni iterata CG
% interna e ogni tentativo di backtracking) e produce, per ogni
% iterazione esterna k:
%   - uno STORYBOARD 2D: un pannello per ogni decisione dell'algoritmo
%     (direzione iniziale -grad, ogni step CG con test di curvatura/eta,
%     direzione finale, Armijo con curva reale phi(alpha))
%   - una VISTA 3D del percorso CG sulla superficie di f
%
% Tutti i grafici finiscono in graphs_broyden31/n2/trace/...

clear; clc; close all;

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbarfun, rfun, xstarfun] = problem_broyden31();

seed    = 346710; % <-- UPDATE THIS WITH YOUR TEAM'S MINIMUM ID[cite: 2]
tolgrad = 1e-6;
n = 2;

% --- STESSA generazione di X0 usata in run_experiments_31.m ---
rng(seed + n, 'twister');
xb = xbarfun(n);
X0 = [xb, xb + (2*rand(n,5) - 1)];

s = 4; % starting point richiesto
x0 = X0(:, s);
xstar_n = xstarfun(n);

fprintf('Starting point %d (n=%d, exact): x0 = [%.6f, %.6f]\n', s, n, x0(1), x0(2));

% --- STESSI parametri truncated usati in run_experiments_31.m ---
params_truncated.kmax   = 20;
params_truncated.c1     = 1e-4;
params_truncated.rho    = 0.3;
params_truncated.btmax  = 10;
params_truncated.max_cg = 20;

[xk, fk, gn, k, xseq, btseq, pks, inner_iters, trace] = ...
    truncated_newton_method_diagnostic(x0, f, grad_exact, hess_exact, ...
        params_truncated.kmax, tolgrad, params_truncated.c1, ...
        params_truncated.rho, params_truncated.btmax, params_truncated.max_cg);

fprintf('Convergenza: %d iterazioni, ||grad||=%.3e, successo=%d\n', k, gn, gn < tolgrad);

for i = 1:numel(trace)
    fprintf(['iter %2d | eta=%.3e | CG=%2d (%-9s) | alpha=%.4f | bt=%2d | ' ...
             '||pk||=%.3e | f: %.6e -> %.6e\n'], ...
        i, trace(i).eta, trace(i).cg_iters, trace(i).cg_exit, ...
        trace(i).alpha_final, trace(i).bt_steps, trace(i).pk_norm, ...
        trace(i).fk, trace(i).fk_new);
end

% --- grafico d'insieme (contour + traiettoria completa) ---
%outdir_overview = fullfile('graphs_broyden31', 'n2', 'trace');
%plot_algorithm_trace(f, trace, x0, xstar_n, 'TruncatedNewton_start2', 'Broyden31', outdir_overview);

% --- storyboard passo-passo + vista 3D, PER OGNI iterazione esterna ---
outdir_story = fullfile('graphs_broyden31', 'n2', 'trace', 'storyboard');
%outdir_3d    = fullfile('graphs_broyden31', 'n2', 'trace', '3d');

for kk = 1:numel(trace)
    plot_cg_storyboard(f, grad_exact, trace(kk), kk, 'TruncatedNewton_start2', 'Broyden31', outdir_story);
    %plot_cg_3d(f, trace(kk), kk, 'TruncatedNewton_start2', 'Broyden31', outdir_3d);
    close all; % chiude i pannelli dello storyboard prima di passare all'iterazione successiva
end

disp('Grafico d''insieme salvato in graphs_broyden31/n2/trace/');
disp('Storyboard passo-passo salvato in graphs_broyden31/n2/trace/storyboard/iterNN/');
disp('Vista 3D salvata in graphs_broyden31/n2/trace/3d/');
