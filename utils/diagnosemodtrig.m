clear; clc; close all;

% ============================================================
% Diagnostica del PRIMO PASSO di Modified Newton su Trig16 (n=2),
% estesa a exact/case1/case2, per i 6 starting point standard.
%
% Vedi la versione originale (solo exact) per la spiegazione completa
% della logica di correzione replicata. Qui in piu' si esplora come
% l'amplificazione del primo passo (lambda_min(Bk) vicino a zero) sia
% influenzata dal rumore delle differenze finite sull'Hessiana.
% ============================================================

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbar_gen] = problem_trig16();

n = 2;
seed = 346710;
beta = 1;       % stesso beta usato in run_trig16_experiments.m
tol_diag = 1e-6;   % stessa tolleranza usata dentro modified_newton_method

k_list  = [4, 8, 12];
fdtypes = [1, 2];

rng(seed + n, 'twister');
xb = xbar_gen(n);
X0 = [xb, xb + (2*rand(n,5) - 1)];

fprintf('\n===============================================\n');
fprintf(' Diagnostica primo passo - Modified Newton - Trig16 (n=2)\n');
fprintf(' beta = %.1e\n', beta);
fprintf('===============================================\n\n');

fprintf('%-6s %-6s %-4s %-4s | %-14s | %-10s | %-16s | %-12s | %-12s | %-12s\n', ...
    'start', 'dm', 'k', 'type', 'lambda_min H0', 'tau_0', 'lambda_min Bk', '||p0||', '||grad(x0)||', 'amplific.');

results = struct([]);
idx = 0;

for dm = ["exact", "case1", "case2"]

    if dm == "exact"
        k_loop = NaN; type_loop = 0;
    else
        k_loop = k_list; type_loop = fdtypes;
    end

    for kk = k_loop
        for type = type_loop

            switch dm
                case "exact"
                    gradf = grad_exact;
                    hessf = hess_exact;
                case "case1"
                    gradf = grad_exact;
                    hessf = @(x) trig_hess_fd_case1(grad_exact, x, kk, type);
                case "case2"
                    gradf = @(x) trig_fd_case2_grad_only(x, kk, type);
                    hessf = @(x) trig_fd_case2_hess_only(x, kk, type);
            end

            for s = 1:6
                x0 = X0(:, s);

                g0 = gradf(x0);
                H0 = full(hessf(x0));

                diagH0 = diag(H0);
                isPositive = all(diagH0 > tol_diag);
                if isPositive
                    tau_0 = 0;
                else
                    tau_0 = beta - min(diagH0);
                end

                maxit = 200;
                tauk = zeros(maxit+1,1);
                tauk(1) = tau_0;
                for j = 1:maxit
                    Bk = H0 + tauk(j)*eye(n);
                    [~, flag] = chol(Bk);
                    if flag == 0
                        tau_0 = tauk(j);
                        break;
                    else
                        tauk(j+1) = max(beta, 2*tauk(j));
                    end
                end
                Bk = H0 + tau_0*eye(n);

                lambda_min_H0 = min(eig(H0));
                lambda_min_Bk = min(eig(Bk));

                p0 = Bk \ (-g0);

                norm_p0    = norm(p0);
                norm_grad0 = norm(g0);
                amplification = norm_p0 / norm_grad0;

                if isnan(kk)
                    kstr = "n/a"; typestr = "n/a";
                else
                    kstr = string(kk);
                    if type == 1, typestr = "h"; else, typestr = "hi"; end
                end

                fprintf('%-6d %-6s %-4s %-4s | %-14.4e | %-10.4e | %-16.4e | %-12.4e | %-12.4e | %-12.2f\n', ...
                    s, dm, kstr, typestr, lambda_min_H0, tau_0, lambda_min_Bk, norm_p0, norm_grad0, amplification);

                idx = idx + 1;
                results(idx).start = s;
                results(idx).dm = char(dm);
                results(idx).k = kk;
                results(idx).type = type;
                results(idx).lambda_min_H0 = lambda_min_H0;
                results(idx).tau_0 = tau_0;
                results(idx).lambda_min_Bk = lambda_min_Bk;
                results(idx).norm_p0 = norm_p0;
                results(idx).norm_grad0 = norm_grad0;
                results(idx).amplification = amplification;
            end
        end
    end
end

fprintf('\n===============================================\n');
fprintf(' "amplific." = ||p0|| / ||grad(x0)||. Confrontare la colonna\n');
fprintf(' per dm=exact vs case1/case2 allo stesso start: se il rumore\n');
fprintf(' FD altera lambda_min(H0) vicino a x0, l''amplificazione del\n');
fprintf(' primo passo puo'' risultare artificialmente piu'' o meno\n');
fprintf(' marcata rispetto al caso esatto.\n');
fprintf('===============================================\n');

save('trig16_first_step_modified_alldm.mat', 'results');
disp('Salvato in trig16_first_step_modified_alldm.mat');