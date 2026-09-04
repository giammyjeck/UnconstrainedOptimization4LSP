clear; clc; close all;

% ============================================================
% Diagnostica del PRIMO PASSO di Truncated Newton su Trig16 (n=2),
% exact/case1/case2, per i 6 starting point standard.
%
% A differenza della versione per Modified Newton, qui NON si
% re-implementa la logica del CG interno (rischio di introdurre
% disallineamenti sottili rispetto all'algoritmo vero): si chiama
% DIRETTAMENTE truncated_newton_method con kmax=1, cosi' si ottiene
% esattamente il primo passo prodotto dall'algoritmo reale, fedele al
% 100% (compreso il test di curvatura negativa, l'eventuale fallback a
% steepest descent, e il criterio di arresto interno eta_k).
%
% Colonne riportate:
%   lambda_min H0   : autovalore minimo dell'Hessiana (esatta o FD) in
%                      x0, per confronto diretto con la diagnostica di
%                      Modified Newton
%   inner_iters      : numero di iterazioni CG usate al primo passo.
%                      0 significa che la curvatura negativa e' stata
%                      rilevata GIA' alla prima iterazione (j=0): il
%                      passo e' il fallback -grad(x0), cioe' steepest
%                      descent puro (vedi modified_newton_method.m,
%                      commento su questo stesso comportamento)
%   eta_k            : tolleranza di arresto interna del CG a questa
%                      iterazione (Dembo-Eisenstat-Steihaug forcing term)
%   ||p0||           : lunghezza del primo passo prodotto
%   ||grad(x0)||     : per confronto diretto con Modified Newton
%   amplific.        : ||p0|| / ||grad(x0)||. Ci si aspetta vicino a 1
%                      quando inner_iters=0 (il passo E' -grad(x0)),
%                      a differenza di Modified Newton dove anche in
%                      quel caso l'amplificazione puo' essere grande
%                      (perche' li' si risolve comunque un sistema
%                      lineare con Bk vicino a singolare, non si fa un
%                      semplice fallback a steepest descent).
% ============================================================

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbar_gen] = problem_trig16();

n = 2;
seed = 346710;

% stessi parametri di params_truncated in run_trig16_experiments.m
c1    = 1e-4;
rho   = 0.8;
btmax = 50;
max_cg = 1000;
tolgrad = 1e-6;

k_list  = [4, 8, 12];
fdtypes = [1, 2];

rng(seed + n, 'twister');
xb = xbar_gen(n);
X0 = [xb, xb + (2*rand(n,5) - 1)];

fprintf('\n===============================================\n');
fprintf(' Diagnostica primo passo - Truncated Newton - Trig16 (n=2)\n');
fprintf('===============================================\n\n');

fprintf('%-6s %-6s %-4s %-4s | %-14s | %-12s | %-12s | %-12s | %-12s | %-12s\n', ...
    'start', 'dm', 'k', 'type', 'lambda_min H0', 'inner_iters', 'eta_k', '||p0||', '||grad(x0)||', 'amplific.');

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
                lambda_min_H0 = min(eig(H0));
                eta_k = min(0.5, sqrt(norm(g0))) * norm(g0);

                % --- chiamata diretta all'algoritmo reale, kmax=1 ---
                try
                    [~, ~, ~, ~, ~, ~, pks, inner_iters] = truncated_newton_method( ...
                        x0, f, gradf, hessf, 1, tolgrad, c1, rho, btmax, max_cg);
                    p0 = pks(:,1);
                    j_cg = inner_iters(1);
                catch ME
                    warning('Truncated fallito (dm=%s,k=%g,type=%d,s=%d): %s', dm, kk, type, s, ME.message);
                    p0 = nan(n,1);
                    j_cg = NaN;
                end

                norm_p0    = norm(p0);
                norm_grad0 = norm(g0);
                amplification = norm_p0 / norm_grad0;

                if isnan(kk)
                    kstr = "n/a"; typestr = "n/a";
                else
                    kstr = string(kk);
                    if type == 1, typestr = "h"; else, typestr = "hi"; end
                end

                fprintf('%-6d %-6s %-4s %-4s | %-14.4e | %-12d | %-12.4e | %-12.4e | %-12.4e | %-12.2f\n', ...
                    s, dm, kstr, typestr, lambda_min_H0, j_cg, eta_k, norm_p0, norm_grad0, amplification);

                idx = idx + 1;
                results(idx).start = s;
                results(idx).dm = char(dm);
                results(idx).k = kk;
                results(idx).type = type;
                results(idx).lambda_min_H0 = lambda_min_H0;
                results(idx).inner_iters = j_cg;
                results(idx).eta_k = eta_k;
                results(idx).norm_p0 = norm_p0;
                results(idx).norm_grad0 = norm_grad0;
                results(idx).amplification = amplification;
            end
        end
    end
end

fprintf('\n===============================================\n');
fprintf(' inner_iters=0 -> curvatura negativa rilevata subito, il\n');
fprintf(' passo e'' il fallback -grad(x0) (steepest descent puro):\n');
fprintf(' ci si aspetta amplific.~1 in quei casi. Se invece inner_iters\n');
fprintf(' > 0 e amplific. si discosta molto da 1, il CG ha comunque\n');
fprintf(' costruito un passo diverso da steepest descent puro.\n');
fprintf('===============================================\n');

save('trig16_first_step_truncated_alldm.mat', 'results');
disp('Salvato in trig16_first_step_truncated_alldm.mat');