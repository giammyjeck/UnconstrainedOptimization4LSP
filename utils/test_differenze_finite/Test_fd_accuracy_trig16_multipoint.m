clear; clc; close all;
 
% ============================================================
% Test di accuratezza delle differenze finite per trig16 su PIU' PUNTI
% del dominio (Monte Carlo), invece di un singolo x_test.
%
% Per ogni combinazione (k, type) campiona N_test punti random nella
% stessa regione usata da run_trig16_experiments.m (xb + spostamento
% random), calcola l'errore relativo di gradiente/Hessiana in ciascun
% punto, e riporta min/media/max sui punti campionati: cosi' si vede
% non solo l'errore in un punto "fortunato" ma la sua variabilita' sul
% dominio tipico incontrato dall'ottimizzatore.
%
% Due metriche per l'Hessiana:
%   - err_fro : ||H_fd - H_ex||_F / ||H_ex||_F        (errore globale/medio)
%   - err_max : max_i |H_fd_ii - H_ex_ii| / |H_ex_ii|  (caso peggiore, elemento)
% La Frobenius puo' "nascondere" un singolo elemento esploso per
% cancellazione se il resto della matrice ha norma grande; la max
% elementwise cattura invece i punti caldi locali.
% ============================================================
 
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));
 
[f, grad_exact, hess_exact, xbar_gen] = problem_trig16(); %#ok<ASGLU>
 
n = 1e4;
N_test = 30;               % numero di punti random campionati
disp_scale = 0.3;           % ampiezza dello spostamento random attorno a xb
 
k_list     = [4, 8, 12];
type_list  = [1, 2];
type_names = ["h (costante)", "hi (relativo)"];
 
rng(346710);
xb = xbar_gen(n);
X_test = xb + disp_scale*randn(n, N_test);   % N_test punti di test
 
fprintf('\n===============================================\n');
fprintf(' Test FD multi-punto - trig16\n');
fprintf(' n = %d,  N_test = %d punti\n', n, N_test);
fprintf('===============================================\n\n');
 
 
% ---------------- CASE 1: solo Hessiana FD ----------------
fprintf('--- CASE 1: gradiente esatto, Hessiana FD ---\n');
fprintf('%-6s %-16s | %-32s | %-32s\n', 'k', 'type', ...
    'err_fro  (min/media/max)', 'err_max  (min/media/max)');
 
for kk = k_list
    for tt = type_list
        err_fro = zeros(N_test,1);
        err_max = zeros(N_test,1);
 
        for p = 1:N_test
            x_test = X_test(:,p);
            g_ex = grad_exact(x_test);
            H_ex = hess_exact(x_test);
 
            H_fd = trig_hess_fd_case1(grad_exact, x_test, kk, tt);
 
            err_fro(p) = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
            d_ex = full(diag(H_ex)); d_fd = full(diag(H_fd));
            err_max(p) = max(abs(d_fd - d_ex) ./ max(abs(d_ex), eps));
        end
 
        fprintf('%-6d %-16s | %8.2e / %8.2e / %8.2e | %8.2e / %8.2e / %8.2e\n', ...
            kk, type_names(tt), ...
            min(err_fro), mean(err_fro), max(err_fro), ...
            min(err_max), mean(err_max), max(err_max));
    end
end
 
 
% ---------------- CASE 2: gradiente e Hessiana FD ----------------
fprintf('\n--- CASE 2: gradiente FD, Hessiana FD ---\n');
fprintf('%-6s %-16s | %-32s | %-32s\n', 'k', 'type', ...
    'err_grad (min/media/max)', 'err_H_fro (min/media/max)');
 
for kk = k_list
    for tt = type_list
        err_g   = zeros(N_test,1);
        err_fro = zeros(N_test,1);
 
        for p = 1:N_test
            x_test = X_test(:,p);
            g_ex = grad_exact(x_test);
            H_ex = hess_exact(x_test);
 
            g_fd = trig_fd_case2_grad_only(x_test, kk, tt);
            H_fd = trig_fd_case2_hess_only(x_test, kk, tt);
 
            err_g(p)   = norm(g_fd - g_ex) / norm(g_ex);
            err_fro(p) = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
        end
 
        fprintf('%-6d %-16s | %8.2e / %8.2e / %8.2e | %8.2e / %8.2e / %8.2e\n', ...
            kk, type_names(tt), ...
            min(err_g), mean(err_g), max(err_g), ...
            min(err_fro), mean(err_fro), max(err_fro));
    end
end
 
fprintf('\n===============================================\n');
fprintf(' Riferimento: tolgrad tipicamente usato = 1e-6\n');
fprintf('===============================================\n');
 
 
% ============================================================
% Boxplot: distribuzione dell'errore Hessiana (case1) sui punti
% campionati, per ciascun k (type = costante). Utile per vedere a
% colpo d'occhio la dispersione, non solo la media.
% ============================================================
figure('Color','w');
err_matrix = zeros(N_test, numel(k_list));
for i = 1:numel(k_list)
    kk = k_list(i);
    for p = 1:N_test
        x_test = X_test(:,p);
        H_ex = hess_exact(x_test);
        H_fd = trig_hess_fd_case1(grad_exact, x_test, kk, 1);
        err_matrix(p,i) = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
    end
end
boxplot(err_matrix, 'Labels', string(k_list));
set(gca, 'YScale', 'log');
grid on;
xlabel('k  (h = 10^{-k})');
ylabel('errore relativo Frobenius Hessiana (case1)');
title('Trig16 case1: dispersione errore su N\_test punti del dominio');
 