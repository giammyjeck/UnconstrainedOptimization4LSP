% Obiettivo: Modified Newton con derivate TUTTE numeriche (caso realistico).
%   - BASELINE: grad exact + Hess exact
%   - CASE 2 : grad FD + Hess numerica (h / hi), k=4,8,12
%
% Scelte "robuste" per ridurre rumore FD:
%   * Trig16: separabile -> uso tfun (contribuzioni) e FD centrali:
%       - g_i ≈ (t_i(x_i+h_i) - t_i(x_i-h_i)) / (2 h_i)
%       - H diagonale: d2_i ≈ (t_i(x_i+h_i) - 2 t_i(x_i) + t_i(x_i-h_i))/h_i^2
%
%   * Broyden31: least squares F=1/2||r||^2
%       - stimo J con FD centrali su r(x), sfruttando banda bwJ=1 e coloring
%       - g = J' r
%       - H = J' J  (Gauss–Newton, PSD, molto più stabile numericamente)
%       - cache: per lo stesso x, grad/hess condividono la stessa J
%
% Success = (||grad f(x_k)|| < tolgrad)
% =========================================================================

clear; clc; close all;
addpath(genpath("C:\Users\Utente\Desktop\Corsi\Numerical optimization for large scale problems and Stochastic Optimization\NumericalO4LSP\main"));


seed = 346710;
rng(seed,"twister");

n_list  = [2, 1e3, 1e4, 1e5];
k_list  = [4, 8, 12];
modes   = ["h","hi"];      % h: 10^-k ; hi: 10^-k * max(|x_i|,1)

% --- parametri Modified Newton ---
kmax    =50;
tolgrad = 1e-6;
c1      = 1e-4;
rho     = 0.5;
btmax   = 10;
beta    = 1e-2;
%max_cg   = 5;

% --- problemi ---
%probs  = {@problem_broyden31};
%pnames = ["problem_broyden31"];

probs  = {@problem_trig16};
pnames = ["problem_trig16"];


outdir = "out_task3_case2_success";
if ~exist(outdir,"dir"), mkdir(outdir); end

results = struct();

for p = 1:numel(probs)

    % Qui assumiamo il tuo formato (come mi hai detto):
    % [f, grad_exact, hess_exact, xbarfun, rfun]
    [f, grad_exact, hess_exact, xbarfun, rfun] = probs{p}();
    pname = pnames(p);

    fprintf("\n==============================\n");
    fprintf("TASK3 CASE2 SUCCESS: %s\n", pname);
    fprintf("==============================\n");

    results.(pname) = struct();

    for n = n_list
        fprintf("\n--- n = %d ---\n", n);

        rng(seed + 1000*p + n, "twister");
        xbar = xbarfun(n);
        X0   = [xbar, xbar + (2*rand(n,5)-1)];

        dim_field = sprintf("n%d", n);
        results.(pname).(dim_field) = struct();

        store = (n == 2);

        % =========================
        % BASELINE (EXACT)
        % =========================
        fprintf("  BASELINE (exact):\n");
        res_exact = run_6starts_success(store, X0, f, grad_exact, hess_exact, ...
            kmax, tolgrad, c1, rho, btmax, beta);
        results.(pname).(dim_field).exact = res_exact;


        % =========================
        % CASE 2: grad FD + Hess numerica
        % =========================
        for kk = 1:numel(k_list)
            kfd = k_list(kk);

            for mm = 1:numel(modes)
                mode = char(modes(mm));
                tag  = sprintf("case2_k%d_%s", kfd, mode);

                fprintf("  %s:\n", tag);

                if pname == "problem_trig16"
                    % Nel tuo setup, rfun per Trig è la tfun (contribuzioni separabili)
                    tfun   = rfun;

                    grad_fd = @(x) trig_fd_grad_central_from_tfun(tfun, x, kfd, mode);
                    hess_fd = @(x) trig_fd_hess_diag_central_from_tfun(tfun, x, kfd, mode);

                else
                    % Broyden: residual rfun(x), stimiamo J bandata e usiamo GN
                    bwJ = 1;
                    [grad_fd, hess_fd] = make_broyden_fd_oracles_cached_GN(rfun, kfd, mode, bwJ);
                end

                res = run_6starts_success(store, X0, f, grad_fd, hess_fd, ...
                    kmax, tolgrad, c1, rho, btmax, beta);

                results.(pname).(dim_field).(tag) = res;
            end
        end
    end
end

save(fullfile(outdir,"results_case2_success.mat"),"results");
disp("Fatto. Salvato in: " + fullfile(outdir,"results_case2_success.mat"));

