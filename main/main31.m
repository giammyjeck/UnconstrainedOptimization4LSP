%% Main Script for Problem 31
clear; clc; close all;

seed = 346710; 
rng(seed);
addpath(genpath(pwd))

% Defining the problem
[f, gradf, hessf, xbar_gen] = problem_broyden31();

% Parameters definition
dimensions = [2, 10^3, 10^4, 10^5]; 
kmax = 50;
tolgrad = 1e-6;
c1 = 1e-3;      % Standard Armijo parameter
rho = 0.3;      % Backtracking contraction factor
btmax = 5;
max_cg = 500;   % Max inner iterations for the conjugate gradient solving method in the truncated one
beta = 1e-2;

% Data structures to store results 
results = struct();
results.truncated = struct();
results.modified = struct();

for i = 1:length(dimensions) % Loop on the problem dimension
    n = dimensions(i);
    fprintf('\n--- Test dimension n = %d (Broyden 31) ---\n', n);

    fprintf('%-10s | %-10s | %-10s | %-8s | %-6s | %-10s | %-8s\n', ...
        'start.pt', 'grad.norm', 'iters/max', 'success', 'flag', 'rate(exp)', 'time');
    fprintf('-----------------------------------------------------------------------------------------\n');
    
    % Starting point suggested by the literature + 5 random
    % starting points in the hyper-cube [xbar-1, xbar+1]
    x0_standard = xbar_gen(n);
    x0_random = (x0_standard - 1) + 2 * rand(n, 5);
    % Combine all starting points (1 standard + 5 random) into a 6x6 matrix
    all_x0 = [x0_standard, x0_random];

    n_success = 0; sum_gnorm = 0; sum_iters = 0; sum_time = 0; sum_rate = 0;
    
    for s = 1:6 % Loop on the 6 starting point
        x0_curr = all_x0(:, s);
        label = sprintf('n%d_pt%d', n, s);
        
        % Truncated Newton
        tstart = tic;
        [xk_tn, fk_tn, gnorm_tn, k_tn, xseq_tn, btseq_tn, pks_tn, inner_tn] = truncated_newton_method( x0_curr,f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, max_cg);
        t_tn = toc(tstart);

        rate_tn = estimate_rate(xseq_tn, gradf);
        flag_tn = derive_flag(k_tn, gnorm_tn, kmax);
        results.truncated.(label) = pack_result(t_tn, k_tn, gnorm_tn, xseq_tn, rate_tn, flag_tn);

        % Modified Newton
        tstart = tic;
        [xk_md,fk_md,gnorm_md,k_md,xseq_md,btseq_md,alphas_md,gradfk_seq,fk_seq,tau_md,pks_md] = modified_newton_method(x0_curr, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta);

        t_md = toc(tstart);

        rate_md = estimate_rate(xseq_md, gradf);
        flag_md = derive_flag(k_md, gnorm_md, kmax);
        results.modified.(label) = pack_result(t_md, k_md, gnorm_md, xseq_md, rate_md, flag_md);
    end
end
% --- Visualizzazione: chiama la funzione che produce tabelle e grafici per ogni metodo ---
display_results(results.truncated, 'Truncated Newton', dimensions, kmax, tolgrad);
display_results(results.modified,  'Modified Newton',  dimensions, kmax, tolgrad);

%%    
plot_results(results.truncated, gradf, dimensions, tolgrad, 'Truncated Newton');
plot_results(results.modified,  gradf, dimensions, tolgrad, 'Modified Newton');

outputs(results, f, gradf, "Problem31");

%% --- HELPER FUNCTIONS ---

function r = pack_result(time, iters, gnorm, xseq, rate, flag)
    r.time = time;
    r.iters = iters;
    r.gnorm = gnorm;
    r.xseq = xseq;
    r.rate = rate;
    r.flag = flag;
end

function flag = derive_flag(k, gnorm, kmax)
    if k >= kmax
        flag = 'maxit';
    else
        if gnorm < 1e-6
            flag = '-';
        else
            flag = 'ls_fail';
        end
    end
end
