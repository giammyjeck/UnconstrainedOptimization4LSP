% This script evaluates the performance of the Modified Newton method across 
% different scales (n up to 10^5) using both exact and numerical derivatives. 
% It automates a comparative analysis of Finite Difference (FD) schemes 
% by testing various step sizes (k=4, 8, 12) and adaptive strategies.

clear; clc; close all;
addpath(genpath("C:\Users\Utente\Desktop\Corsi\Numerical optimization for large scale problems and Stochastic Optimization\NumericalO4LSP\main"));


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

% Output initialization.
outdir = "out_fd_grad_hess";
if ~exist(outdir,"dir"), mkdir(outdir); end
results = struct();


for p = 1:numel(probs)

    % Load problem data.
    [f, grad_exact, hess_exact, xbarfun, rfun] = probs{p}();
    pname = pnames(p);


    fprintf("FINITE DIFFERENCES GRAD + HESS PROCESSING: %s\n", pname);
    results.(pname) = struct();

    for n = n_list
        fprintf("\n--- n = %d ---\n", n);
        
        % Generating starting points.
        rng(seed + 1000*p + n, "twister");
        xbar = xbarfun(n);
        X0   = [xbar, xbar + (2*rand(n,5)-1)];

        dim_field = sprintf("n%d", n);
        results.(pname).(dim_field) = struct();
        
        % Only storing information about 2-dim.
        store = (n == 2);

        % CASE 1: Solving problem p with exact derivatives.
        fprintf("CASE 1 (exact):\n");
        res_exact = run_simulation(store, X0, f, grad_exact, hess_exact, ...
            kmax, tolgrad, c1, rho, btmax, beta);
        results.(pname).(dim_field).exact = res_exact;

        % CASE 2: Solving problem p with exact derivatives.
        for kk = 1:numel(k_list) % Loop on possible k values.
            kfd = k_list(kk); 

            for mm = 1:numel(modes) % Loop on possible ks (fixed or variable).
                mode = char(modes(mm));
                tag  = sprintf("case2_k%d_%s", kfd, mode);
                fprintf("  > Running FD: %s (k=%d, mode=%s)...\n", tag, kfd, mode);
                

                % Logic for Problem16: Since each component of the function depends 
                % solely on its corresponding variable $x_i$, the resulting Hessian
                % matrix is diagonal. This structural property allows for the computation
                % of only $n$ second-order derivatives instead of $n^2$, significantly
                % reducing the computational overhead for large-scale instances."
                if pname == "problem16_Trig"

                    tfun   = rfun;
                    
                    % FD_16: gradient and hessian
                    grad_fd = @(x) trig_fd_grad_central_from_tfun(tfun, x, kfd, mode);
                    hess_fd = @(x) trig_fd_hess_diag_central_from_tfun(tfun, x, kfd, mode);

                else
                    % Logic for Problem31: since the Jacobian is
                    % tridiagonal we chose to approximate just that in order to 
                    % drastically reduce function evaluations. The Hessian is 
                    % then estimated via Gauss-Newton (H approx J'J), ensuring 
                    % a symmetric positive semi-definite matrix and improving numerical stability.
                    
                    bwJ = 1;
                    % FD_31: gradient and hessian
                    [grad_fd, hess_fd] = make_broyden_fd_oracle(rfun, kfd, mode, bwJ);
                end

                res = run_simulation(store, X0, f, grad_fd, hess_fd, ...
                    kmax, tolgrad, c1, rho, btmax, beta);

                results.(pname).(dim_field).(tag) = res;
            end
        end
    end
end

save(fullfile(outdir,"results_fd_grad_hess.mat"),"results");
disp("Done. Saved in: " + fullfile(outdir,"results_fd_grad_hess.mat"));

