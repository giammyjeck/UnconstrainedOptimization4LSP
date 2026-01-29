% =========================================================================
% TASK 3 - CASE 1 (con SUCCESS)
%
% Obiettivo: capire se Modified Newton converge quando SOLO la Hessiana è FD.
%   - BASELINE: grad exact + Hess exact
%   - CASE 1 : grad exact + Hess FD (h / hi), k=4,8,12
%
% Success = (||grad f(x_k)|| < tolgrad)
%
% Nota: Hess FD bandata con "coloring" 1D:
%   - bw=0  (Trig16: Hess diagonale)
%   - bw=2  (Broyden31: Hess pentadiagonale)
% In questo modo non facciamo n valutazioni di gradiente, ma (2*bw+1).
% =========================================================================

clear; clc; close all;
addpath(genpath("C:\Users\Utente\Desktop\Corsi\Numerical optimization for large scale problems and Stochastic Optimization\NumericalO4LSP\main"));

% --- seed richiesto dal progetto ---
seed = 346710;
rng(seed,"twister");

% --- dimensioni e step FD ---
n_list  = [2, 1e3, 1e4, 1e5];
k_list  = [4, 8, 12];
modes   = ["h","hi"];      % h: 10^-k ; hi: 10^-k * max(|x_i|,1)

% --- parametri Modified Newton ---
kmax    = 50;
tolgrad = 1e-6;
c1      = 1e-4;
rho     = 0.3;
btmax   = 10;
beta    = 1e-2;

% --- problemi ---
probs  = {@problem_trig16, @problem_broyden31};
pnames = ["problem_trig16","problem_broyden31"];

% half-bandwidth Hessiana (per FD banded)
bwH = [0, 2];

% 0 ho solo la diagonale, 1 è tridiagonale  e 3 è tridiagonale 

% --- output ---
outdir = "out_task3_case1_success";
if ~exist(outdir,"dir"), mkdir(outdir); end

results = struct();

for p = 1:numel(probs)

    % mi aspetto che problem_* ritorni almeno:
    % [f, grad_exact, hess_exact, xbarfun, rfun]  (rfun qui ignorata)
    [f, grad_exact, hess_exact, xbarfun] = probs{p}();
    pname = pnames(p);
    bw    = bwH(p);

    fprintf("\n==============================\n");
    fprintf("TASK3 CASE1 SUCCESS: %s (bwH=%d)\n", pname, bw);
    fprintf("==============================\n");

    results.(pname) = struct();

    for n = n_list
        fprintf("\n--- n = %d ---\n", n);

        % stessi random start per confronto pulito
        rng(seed + 1000*p + n, "twister");

        xbar = xbarfun(n);
        X0   = [xbar, xbar + (2*rand(n,5)-1)];  % 6 start: xbar + 5 random

        dim_field = sprintf("n%d", n);
        results.(pname).(dim_field) = struct();

        % Per n grande, NON vogliamo salvare xseq/pks (se il tuo MN salva storia)
        store = (n == 2);

        % =========================
        % BASELINE (EXACT)
        % =========================
        fprintf("  BASELINE (exact):\n");
        res_exact = run_6starts_success(store, X0, f, grad_exact, hess_exact, ...
            kmax, tolgrad, c1, rho, btmax, beta);
        results.(pname).(dim_field).exact = res_exact;

        % =========================
        % CASE 1: grad exact + Hess FD bandata
        % =========================
        for kk = 1:numel(k_list)
            kfd = k_list(kk);

            for mm = 1:numel(modes)
                mode = char(modes(mm));
                tag  = sprintf("case1_k%d_%s", kfd, mode);

                fprintf("  %s:\n", tag);

                % Hess FD = FD del gradiente ESATTO, ma SOLO nella banda
                Hfd = @(x) hess_fd_from_grad_banded(grad_exact, x, kfd, mode, bw);

                res = run_6starts_success(store, X0, f, grad_exact, Hfd, ...
                    kmax, tolgrad, c1, rho, btmax, beta);

                results.(pname).(dim_field).(tag) = res;
            end
        end
    end
end

save(fullfile(outdir,"results_case1_success.mat"),"results");
disp("Fatto. Salvato in: " + fullfile(outdir,"results_case1_success.mat"));
