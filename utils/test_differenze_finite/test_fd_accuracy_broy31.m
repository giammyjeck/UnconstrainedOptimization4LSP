clear; clc; close all;
 
% ============================================================
% Test di accuratezza delle differenze finite per Broyden31.
%
% Stessa logica del test analogo per trig16: confronta gradiente/
% Hessiana esatti con le loro approssimazioni FD (case1: solo
% Hessiana FD, case2: gradiente e Hessiana entrambi FD), per diversi
% valori di k (passo h = 10^-k) e type (1 = costante, 2 = relativo),
% su un punto di test lontano dal minimo.
%
% Serve a capire, INDIPENDENTEMENTE dall'ottimizzatore, se le FD sono
% accurate abbastanza da permettere gradnorm < tolgrad. Se l'errore
% relativo delle FD e' gia' piu' grande di tolgrad, nessuna
% configurazione di modified/truncated Newton potra' mai convergere,
% qualunque siano rho/c1/bt/max_cg.
% ============================================================
 
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));
 
[f, grad_exact, hess_exact, xbar_gen, rfun] = problem_broyden31(); %#ok<ASGLU>
 
n = 1e5;
 
% Punto di test: NON il minimo, ma un punto "tipico" incontrato durante
% l'ottimizzazione, cosi' il test e' rappresentativo di cosa vede
% davvero l'algoritmo (vicino al minimo le derivate FD sono spesso piu'
% "facili" da approssimare bene, quindi testare solo li' sarebbe
% ottimistico).
rng(346710);
x_test = xbar_gen(n) + 0.3*randn(n, 1);
 
k_list    = [4, 8, 12];
type_list = [1, 2];
type_names = ["h (costante)", "hi (relativo)"];
 
g_ex = grad_exact(x_test);
H_ex = hess_exact(x_test);
 
fprintf('\n===============================================\n');
fprintf(' Test accuratezza differenze finite - Broyden31\n');
fprintf(' n = %d,  ||grad_exact|| = %.6e\n', n, norm(g_ex));
fprintf('===============================================\n\n');
 
 
% ---------------- CASE 1: solo Hessiana FD ----------------
fprintf('--- CASE 1: gradiente esatto, Hessiana FD ---\n');
fprintf('%-6s %-16s %-20s\n', 'k', 'type', 'err relativo Hessiana');
 
for kk = k_list
    for tt = type_list
        H_fd = hess_fd_broyden31(grad_exact, x_test, kk, tt);
 
        err_H = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
 
        fprintf('%-6d %-16s %-20.3e\n', kk, type_names(tt), err_H);
    end
end
 
 
% ---------------- CASE 2: gradiente e Hessiana FD ----------------
fprintf('\n--- CASE 2: gradiente FD, Hessiana FD ---\n');
fprintf('%-6s %-16s %-20s %-20s\n', 'k', 'type', 'err relativo grad', 'err relativo Hessiana');
 
for kk = k_list
    for tt = type_list
        g_fd = grad_fd_broyden31(x_test, kk, tt, rfun);
        H_fd = hess_grad_fd_broyden31(x_test, kk, tt, rfun);
 
        err_g = norm(g_fd - g_ex) / norm(g_ex);
        err_H = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
 
        fprintf('%-6d %-16s %-20.3e %-20.3e\n', kk, type_names(tt), err_g, err_H);
    end
end
 
 
fprintf('\n===============================================\n');
fprintf(' Riferimento: tolgrad tipicamente usato = 1e-6\n');
fprintf(' Se err relativo grad >= tolgrad, il rumore delle FD\n');
fprintf(' rende irraggiungibile la convergenza indipendentemente\n');
fprintf(' dai parametri dell''ottimizzatore.\n');
fprintf('===============================================\n');
 
 
% ============================================================
% Grafico: errore relativo vs k, per capire visivamente se
% l'errore migliora o peggiora al crescere di k (cioe' al
% rimpicciolirsi di h = 10^-k). Se peggiora per k grande, siamo
% nel regime dominato da cancellazione numerica, non da
% troncamento.
% ============================================================
figure('Color','w');
hold on; grid on; box on;
 
err_g_h  = zeros(size(k_list));
err_g_hi = zeros(size(k_list));
 
for i = 1:numel(k_list)
    kk = k_list(i);
    g_fd_h  = grad_fd_broyden31(x_test, kk, 1, rfun);
    g_fd_hi = grad_fd_broyden31(x_test, kk, 2, rfun);
    err_g_h(i)  = norm(g_fd_h  - g_ex) / norm(g_ex);
    err_g_hi(i) = norm(g_fd_hi - g_ex) / norm(g_ex);
end
 
semilogy(k_list, err_g_h,  '-o', 'LineWidth', 1.5, 'DisplayName', 'type = h (costante)');
semilogy(k_list, err_g_hi, '-s', 'LineWidth', 1.5, 'DisplayName', 'type = hi (relativo)');
yline(1e-6, '--k', 'tolgrad = 1e-6', 'LabelHorizontalAlignment', 'left');
 
xlabel('k  (h = 10^{-k})');
ylabel('errore relativo sul gradiente');
title('Broyden31: accuratezza del gradiente FD al variare di k');
legend('Location', 'best');
 