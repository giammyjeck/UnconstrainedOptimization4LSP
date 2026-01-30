% This script analyzes the performance of the Modified Newton method where the 
% gradient is exact, while the Hessian is approximated via Finite Differences (FD).
% It evaluates the convergence and scalability (up to n=10^5) across different 
% FD step sizes (k=4, 8, 12) and strategies.

clear; clc; close all;
addpath(genpath("C:\Users\giovannamaccarone\Desktop\NumericalO4LSP\main"));


seed = 346710;
rng(seed,"twister");

% Working with two parameters, one fixed and the other depending on a k
% parameter.
n_list  = [2, 1e3, 1e4, 1e5];
k_list  = [4, 8, 12];
modes   = ["h","hi"];   

% Parameters (Modified Newton)
kmax    = 50;           % Max iterations.
tolgrad = 1e-6;         % Tolerance for stopping criterion.
c1      = 1e-4;         % Armijo sufficient decrease parameter.
rho     = 0.3;          % Backtracking reduction factor.
btmax   = 10;           % Max backtracking steps.
beta    = 1e-2;         % Hessian modification parameter.

% Problem definitions.
probs  = {@problem_trig16, @problem_broyden31};
pnames = ["problem_trig16","problem_broyden31"];


% Half-bandwidth for Hessian FD (0: diagonal, 1: tridiagonal, 2: pentadiagonal)
bwH = [0, 2];

% Output initialization.
outdir = "out_fd_hess";
if ~exist(outdir,"dir"), mkdir(outdir); end
results = struct();


for p = 1:numel(probs)

    % Load problem data.
    [f, grad_exact, hess_exact, xbarfun] = probs{p}();
    pname = pnames(p);
    bw    = bwH(p);

    fprintf("FINITE DIFFERENCES HESS PROCESSING: %s\n", pname);
    results.(pname) = struct();

    for n = n_list
        fprintf("\n--- n = %d ---\n", n);

        % Generating starting points.
        rng(seed + 1000*p + n, "twister");
        xbar = xbarfun(n);
        X0   = [xbar, xbar + (2*rand(n,5)-1)];  % 6 start: xbar + 5 random

        dim_field = sprintf("n%d", n);
        results.(pname).(dim_field) = struct();

        % Only storing information about 2-dim.
        store = (n == 2);

        % CASE 1: Solving problem p with exact derivatives.
        fprintf("CASE 1 (exact):\n");
        res_exact = run_simulation(store, X0, f, grad_exact, hess_exact, ...
            kmax, tolgrad, c1, rho, btmax, beta);
        results.(pname).(dim_field).exact = res_exact;

        % CASE 2: Solving problem p with exact gradient and approximate
        % hessian.
        for kk = 1:numel(k_list) % Loop on possible k values.
            kfd = k_list(kk);

            for mm = 1:numel(modes) % Loop on possible ks (fixed or variable).
                mode = char(modes(mm));
                tag  = sprintf("case1_k%d_%s", kfd, mode);

                fprintf("  %s:\n", tag);

                % Differentiation between problems is handled via the 'bw'
                % variable.
                % If bw = 0 -> Problem 16 since H is diagonal. If bw =
                % 2 -> Problem 31 since H is pentadiagonal.
                Hfd = @(x) hess_fd_from_grad_banded(grad_exact, x, kfd, mode, bw);

                res = run_6starts_success(store, X0, f, grad_exact, Hfd, ...
                    kmax, tolgrad, c1, rho, btmax, beta);

                results.(pname).(dim_field).(tag) = res;
            end
        end
    end
end

save(fullfile(outdir,"results_fd_hess.mat"),"results");
disp("Done. Saved in: " + fullfile(outdir,"results_fd_hess.mat"));
