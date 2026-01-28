%% Main Script for Problem 16
clear; clc; close all;
addpath(genpath(pwd))

student_id = 346710; 
rng(student_id);

% Defining the problem
[f, gradf, hessf, xbar_gen] = problem_trig16();

% Parameters definition
dim = [2, 10^3, 10^4]; % NEED TO ADD 10^5
kmax = 1000;
tolgrad = 1e-6;
c1 = 1e-4;      % Standard Armijo parameter
rho = 0.5;      % Backtracking contraction factor
btmax = 20;
max_cg = 500;   % Max inner iterations for the conjugate gradient solving method in the truncated one
beta = 1e-3; % Correction parameter for the Hessian

dim_idx =0;
% Data structures to store results 
experimentalMatrix = struct(); 

for n = dim % Loop on the problem dimension
    dim_idx = dim_idx + 1;

    fprintf('\n--- Test dimension n = %d (Function 16) ---\n', n);

    fprintf('%-10s | %-10s | %-10s | %-8s | %-6s | %-10s | %-8s\n', ...
        'start.pt', 'grad.norm', 'iters/max', 'success', 'flag', 'rate(exp)', 'time');
    fprintf('------------------------------------------------------------------------------------------\n');
    
    % Starting point suggested by the literature + 5 random
    % starting points in the hyper-cube [xbar-1, xbar+1]
    x0_standard = xbar_gen(n);
    x0_random = (x0_standard - 1) + 2 * rand(n, 5);
    % Combine all starting points (1 standard + 5 random) into a 6x6 matrix
    all_x0 = [x0_standard, x0_random];

    n_success = 0; sum_gnorm = 0; sum_iters = 0; sum_time = 0; sum_rate = 0;

    % Store size information
    experimentalMatrix(dim_idx).n = n;
    experimentalMatrix(dim_idx).runs = struct();

    for s = 1:6 % Loop on the 6 starting point
        x0_curr = all_x0(:, s);
        % point_label = sprintf('n%d_pt%d', n, s);
        
        tic;
        %[xk_tn, fk_tn, gnorm_tn, k_tn, xseq_tn, btseq_tn, pks_tn, inner_tn] = ...
        %    truncated_newton_method(x0_curr, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, max_cg);
        [xk_tn,fk_tn,gnorm_tn,k_tn,xseq_tn,btseq_tn,alphas,gradfk_seq,fk_seq,tau_new] = modified_newton_method(x0_curr,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta);
        time_tn = toc;
        
        %CALCOLO DATI PER TABELLA
        is_success = gnorm_tn < tolgrad;
        success_str = 'no';
        if is_success
            success_str = 'yes';
        end
        
        flag = '-';
        if k_tn >= kmax
            flag = 'maxit';
        end
        % Se si ferma prima di kmax ma il gradiente è alto, è fallito il backtracking
        if ~is_success && k_tn < kmax
            flag = 'ls_fail';
        end
        
        % Stima del rate di convergenza (esponente p)
        rate_exp = estimate_rate(xseq_tn, gradf);
        
        % Storing results
        experimentalMatrix(dim_idx).runs(s).x0 = x0_curr;
        experimentalMatrix(dim_idx).runs(s).xk = xk_tn;
        experimentalMatrix(dim_idx).runs(s).fk = fk_tn;
        experimentalMatrix(dim_idx).runs(s).gradfk_norm = gnorm_tn;
        experimentalMatrix(dim_idx).runs(s).k = k_tn;
        experimentalMatrix(dim_idx).runs(s).time = time_tn;
        experimentalMatrix(dim_idx).runs(s).xseq = xseq_tn;
        experimentalMatrix(dim_idx).runs(s).btseq = btseq_tn;
        %experimentalMatrix(dim_idx).runs(s).inner_it = inner_tn;
        %%experimentalMatrix(dim_idx).runs(s).alphas = alphas;
        %experimentalMatrix(dim_idx).runs(s).pks = pks_tn;

        if is_success
            n_success = n_success + 1;
            sum_gnorm = sum_gnorm + gnorm_tn;
            sum_iters = sum_iters + k_tn;
            sum_time = sum_time + time_tn;
            if ~isnan(rate_exp)
                sum_rate = sum_rate + rate_exp;
            end
        end
        
        % Output riga tabella
        pt_name = 'random'; 
        if s==1
            pt_name = 'x_bar';
        end
        fprintf('%-10s | %-10.2e | %-3d/%-6d | %-8s | %-6s | %-10.2f | %-7.2fs\n', ...
            pt_name, gnorm_tn, k_tn, kmax, success_str, flag, rate_exp, time_tn);
    end
    % --- RIGA MEDIA (AVG SUCCESSES) ---
    if n_success > 0
        fprintf('------------------------------------------------------------------------------------------\n');
        fprintf('%-10s | %-10.2e | %-3d/%-6d | %-8s | %-6s | %-10.2f | %-7.2fs\n', ...
            'Avg (succ)', sum_gnorm/n_success, round(sum_iters/n_success), kmax, '-', '-', sum_rate/n_success, sum_time/n_success);
    end
    fprintf('------------------------------------------------------------------------------------------\n');
end

% --- PLOTTING ---
% Chiama la funzione per generare tutti i grafici richiesti
outputs(experimentalMatrix, f, gradf, "Problem16_Trig");


% %% --- HELPER FUNCTIONS ---
 function p = estimate_rate(xseq, gradf)
     if size(xseq, 2) < 4, p = NaN; return; end
     g3 = norm(gradf(xseq(:, end)));
     g2 = norm(gradf(xseq(:, end-1)));
     g1 = norm(gradf(xseq(:, end-2)));
     if g1 < 1e-14 || g2 < 1e-14 || g3 < 1e-14, p = NaN; return; end
     p = log(g3/g2) / log(g2/g1);
     if p < 0 || p > 3, p = NaN; end
 end
