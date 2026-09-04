clear; clc; close all;
 
% ============================================================
% Test di accuratezza delle differenze finite per Broyden31 su PIU'
% PUNTI del dominio (Monte Carlo), invece di un singolo x_test.
% Vedi test_fd_accuracy_trig16_multipoint.m per la spiegazione completa
% della metodologia e delle metriche usate.
% ============================================================
 
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));
 
[f, grad_exact, hess_exact, xbar_gen, rfun] = problem_broyden31(); %#ok<ASGLU>
 
n = 1e4;              % ridotto rispetto a 1e5 per tenere ragionevole il costo di N_test ripetizioni
N_test = 30;
disp_scale = 0.3;
 
k_list     = [4, 8, 12];
type_list  = [1, 2];
type_names = ["h (costante)", "hi (relativo)"];
 
rng(346710);
xb = xbar_gen(n);
X_test = xb + disp_scale*randn(n, N_test);
 
fprintf('\n===============================================\n');
fprintf(' Test FD multi-punto - Broyden31\n');
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
            H_ex = hess_exact(x_test);
 
            H_fd = hess_fd_broyden31(grad_exact, x_test, kk, tt);
 
            err_fro(p) = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
 
            % Errore max elementwise SOLO sui non-zeri del pattern
            % pentadiagonale (altrimenti max(.../0) esplode fuori banda).
            mask = H_ex ~= 0;
            ex_v = full(H_ex(mask)); fd_v = full(H_fd(mask));
            err_max(p) = max(abs(fd_v - ex_v) ./ max(abs(ex_v), eps));
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
 
            g_fd = grad_fd_broyden31(x_test, kk, tt, rfun);
            H_fd = hess_grad_fd_broyden31(x_test, kk, tt, rfun);
 
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
% Boxplot: dispersione dell'errore Hessiana (case1) sui punti
% campionati, per ciascun k (type = costante).
% ============================================================
figure('Color','w');
err_matrix = zeros(N_test, numel(k_list));
for i = 1:numel(k_list)
    kk = k_list(i);
    for p = 1:N_test
        x_test = X_test(:,p);
        H_ex = hess_exact(x_test);
        H_fd = hess_fd_broyden31(grad_exact, x_test, kk, 1);
        err_matrix(p,i) = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
    end
end
boxplot(err_matrix, 'Labels', string(k_list));
set(gca, 'YScale', 'log');
grid on;
xlabel('k  (h = 10^{-k})');
ylabel('errore relativo Frobenius Hessiana (case1)');
title('Broyden31 case1: dispersione errore su N\_test punti del dominio');
 