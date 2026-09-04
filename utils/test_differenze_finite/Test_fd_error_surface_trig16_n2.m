clear; clc; close all;
 
% ============================================================
% Visualizzazione 3D dell'errore delle differenze finite su TUTTO IL
% DOMINIO (x1,x2), per n=2, invece che in un singolo punto.
%
% Per ogni punto della griglia (x1,x2) si calcola l'errore relativo di
% Frobenius dell'Hessiana FD rispetto a quella esatta, e lo si plotta
% come superficie z = errore(x1,x2). Permette di vedere se l'errore e'
% uniforme sul dominio oppure se ci sono zone (es. vicino a punti dove
% qualche componente e' vicina a 0, rilevante per type=2 "hi") dove le
% FD si comportano molto peggio.
%
% Layout: righe = {case1, case2}, colonne = k in k_list. Un'unica
% figura per ciascun "type" (costante / relativo).
% ============================================================
 
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));
 
[f, grad_exact, hess_exact, xbar_gen] = problem_trig16(); %#ok<ASGLU>
 
n = 2;
k_list = [4, 8, 12];
 
% Dominio coerente con run_trig16_experiments.m: xb = [1;1], punti
% random in xb + [-1,1]^2 -> dominio [0,2] x [0,2].
xb = xbar_gen(n);
n_grid = 100;
x1_vec = linspace(xb(1)-1, xb(1)+1, n_grid);
x2_vec = linspace(xb(2)-1, xb(2)+1, n_grid);
[X1, X2] = meshgrid(x1_vec, x2_vec);
 
for tt = [1, 2]
    if tt == 1, type_name = 'h (costante)'; else, type_name = 'hi (relativo)'; end
 
    figure('Color','w', 'Position', [50 50 1400 700]);
    sgtitle(sprintf('Trig16 (n=2): errore relativo Frobenius Hessiana su dominio, type = %s', type_name));
 
    for i_k = 1:numel(k_list)
        kk = k_list(i_k);
 
        Err1 = zeros(n_grid, n_grid);   % case1
        Err2 = zeros(n_grid, n_grid);   % case2
 
        for a = 1:n_grid
            for b = 1:n_grid
                x_pt = [X1(a,b); X2(a,b)];
                H_ex = hess_exact(x_pt);
 
                H1 = trig_hess_fd_case1(grad_exact, x_pt, kk, tt);
                Err1(a,b) = norm(H1 - H_ex, 'fro') / norm(H_ex, 'fro');
 
                H2 = trig_fd_case2_hess_only(x_pt, kk, tt);
                Err2(a,b) = norm(H2 - H_ex, 'fro') / norm(H_ex, 'fro');
            end
        end
 
        % --- Riga 1: case1 ---
        subplot(2, numel(k_list), i_k);
        surf(X1, X2, log10(Err1), 'EdgeColor', 'none');
        xlabel('x_1'); ylabel('x_2'); zlabel('log_{10}(err)');
        title(sprintf('case1, k=%d', kk));
        colorbar; view(45,30);
 
        % --- Riga 2: case2 ---
        subplot(2, numel(k_list), numel(k_list) + i_k);
        surf(X1, X2, log10(Err2), 'EdgeColor', 'none');
        xlabel('x_1'); ylabel('x_2'); zlabel('log_{10}(err)');
        title(sprintf('case2, k=%d', kk));
        colorbar; view(45,30);
    end
end
