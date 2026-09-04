clear; clc; close all;

% ============================================================
% Diagnostica del PRIMO PASSO di Modified Newton FLIP su Trig16 (n=2),
% estesa a exact/case1/case2, per i 6 starting point standard.
%
% Analogo a trig16_first_step_modified_alldm.m, ma la correzione
% replicata qui e' quella di modified_newton_method_flip.m (flip degli
% autovalori <= tol_diag, con floor eps_floor), NON Hk + tau*I. Non
% c'e' quindi ne' tau_0 ne' un loop di raddoppio: la correzione e'
% "one-shot" sullo spettro di H0.
%
% Come nell'originale, la Hessiana qui e' presa piena (full) e gli
% autovalori con eig(), non tramite il ramo "is_diag" veloce usato
% dentro il solver vero e proprio: a n=2 il costo e' irrilevante e
% cosi' il codice resta piu' leggibile e direttamente confrontabile
% riga per riga con la formula "Q * diag(lambda_mod) * Q'" del solver.
% ============================================================

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbar_gen] = problem_trig16();

n = 2;
seed = 346710;
eps_floor = 1e-8;   % stesso floor usato dentro modified_newton_method_flip.m
tol_diag  = 1e-6;   % stessa soglia di flip usata nel solver (d <= 1e-6 / lambda <= 0 nel ramo non-diag)

k_list  = [4, 8, 12];
fdtypes = [1, 2];

rng(seed + n, 'twister');
xb = xbar_gen(n);
X0 = [xb, xb + (2*rand(n,5) - 1)];

fprintf('\n===============================================\n');
fprintf(' Diagnostica primo passo - Modified Newton FLIP - Trig16 (n=2)\n');
fprintf(' eps_floor = %.1e\n', eps_floor);
fprintf('===============================================\n\n');

fprintf('%-6s %-6s %-4s %-4s | %-14s | %-16s | %-12s | %-12s | %-12s | %-8s\n', ...
    'start', 'dm', 'k', 'type', 'lambda_min H0', 'lambda_min Bk', '||p0||', '||grad(x0)||', 'amplific.', 'flipped');

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

                % --- Replica ESATTA della logica in modified_newton_method_flip.m
                % (ramo non-diagonale: decomposizione spettrale completa,
                % flip degli autovalori <= 0, floor eps_floor). Qui si
                % usa sempre eig() indipendentemente dal fatto che H0
                % sia diagonale, per restare fedeli al ramo "is_diag ==
                % false" del solver e confrontabile con l'analisi tau*I
                % gia' fatta; a n=2 il costo e' comunque nullo.
                [Q, D] = eig(H0);
                lambda = diag(D);
                applied = any(lambda <= 0);
                lambda_mod = lambda;
                lambda_mod(lambda <= 0) = max(abs(lambda(lambda <= 0)), eps_floor);
                Bk = Q * diag(lambda_mod) * Q';

                lambda_min_H0 = min(lambda);
                lambda_min_Bk = min(lambda_mod);

                p0 = -Bk \ g0;

                norm_p0    = norm(p0);
                norm_grad0 = norm(g0);
                amplification = norm_p0 / norm_grad0;

                if isnan(kk)
                    kstr = "n/a"; typestr = "n/a";
                else
                    kstr = string(kk);
                    if type == 1, typestr = "h"; else, typestr = "hi"; end
                end

                fprintf('%-6d %-6s %-4s %-4s | %-14.4e | %-16.4e | %-12.4e | %-12.4e | %-12.2f | %-8d\n', ...
                    s, dm, kstr, typestr, lambda_min_H0, lambda_min_Bk, norm_p0, norm_grad0, amplification, applied);

                idx = idx + 1;
                results(idx).start = s;
                results(idx).dm = char(dm);
                results(idx).k = kk;
                results(idx).type = type;
                results(idx).lambda_min_H0 = lambda_min_H0;
                results(idx).lambda_min_Bk = lambda_min_Bk;
                results(idx).norm_p0 = norm_p0;
                results(idx).norm_grad0 = norm_grad0;
                results(idx).amplification = amplification;
                results(idx).flipped = applied;
            end
        end
    end
end

fprintf('\n===============================================\n');
fprintf(' "amplific." = ||p0|| / ||grad(x0)||. "flipped" = 1 se almeno\n');
fprintf(' un autovalore di H0 e'' stato flippato (<=0) a questo start.\n');
fprintf(' Confrontare con trig16_first_step_modified_alldm.m: a parita''\n');
fprintf(' di (start, dm, k, type), lambda_min_H0 e'' IDENTICO (stessa H0),\n');
fprintf(' quindi ogni differenza in lambda_min_Bk / amplific. e'' dovuta\n');
fprintf(' SOLO al meccanismo di correzione (tau*I vs flip), non al rumore\n');
fprintf(' delle differenze finite.\n');
fprintf('===============================================\n');

save('trig16_first_step_flip_alldm.mat', 'results');
disp('Salvato in trig16_first_step_flip_alldm.mat');