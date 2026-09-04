clear; clc; close all;
 
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));
 
[f, gradf, hessf, xbar_gen] = problem_trig16();
 
n    = 1e5;
seed = 346710;
 
rng(seed + n, 'twister');
xb = xbar_gen(n);
X0 = [xb, xb + (2*rand(n,5) - 1)];
 
kmax    = 200;
tolgrad = 1e-6;
c1      = 1e-4;
rho     = 0.5;
btmax   = 40;
beta    = 1e-2;
 
fprintf('===============================================\n');
fprintf(' Confronto: correzione Hk+tau*I  vs  flipping\n');
fprintf('===============================================\n');
 
xseq_list_orig = cell(1, size(X0,2));
xseq_list_flip = cell(1, size(X0,2));
 
for s = 1:size(X0, 2)
    x0 = X0(:, s);
 
    [xk_orig, fk_orig, gn_orig, k_orig, xseq_orig] = modified_newton_method( ...
        x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta);
 
    [xk_flip, fk_flip, gn_flip, k_flip, xseq_flip, ~, ~, ~, ~, n_flips] = modified_newton_method_flip( ...
        x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax);
 
    xseq_list_orig{s} = xseq_orig;
    xseq_list_flip{s} = xseq_flip;
 
    max_x_orig = max(abs(xseq_orig(:)));
    max_x_flip = max(abs(xseq_flip(:)));
 
    fprintf('\n--- Start %d: x0 = [%.4f, %.4f] ---\n', s, x0(1), x0(2));
    fprintf('  ORIGINALE (Hk+tau*I): k=%2d, f=%.6e, ||grad||=%.2e, max|x| lungo il cammino=%.3e\n', ...
        k_orig, fk_orig, gn_orig, max_x_orig);
    fprintf('  FLIPPING            : k=%2d, f=%.6e, ||grad||=%.2e, max|x| lungo il cammino=%.3e, correzioni=%d\n', ...
        k_flip, fk_flip, gn_flip, max_x_flip, n_flips);
end
 
% --- Contour plot di confronto ---
plot_contour_paths(f, xseq_list_orig, 'ModifiedNewton_tauI', 'Trig16', 'graphs_trig16_flip_compare');
plot_contour_paths(f, xseq_list_flip, 'ModifiedNewton_flip', 'Trig16', 'graphs_trig16_flip_compare');
 
disp('Contour plot salvati in graphs_trig16_flip_compare/');
 