%% ============================================================
% TEST SCRIPT – MODIFIED NEWTON
% Problem: Generalized Broyden Tridiagonal Function
% ============================================================

clear; clc; close all;

%% ================= RANDOM SEED ==============================
rng(346710);

%% ================= PARAMETERS ===============================
kmax    = 1000;
tolgrad = 1e-6;
c1      = 1e-4;
rho     = 0.5;
btmax   = 20;
beta    = 1e-3;

p = 7/3;
n_values = 1e4;

%% ================= LOOP OVER DIMENSIONS =====================
for n = n_values

    fprintf('\n===========================================\n');
    fprintf(' Generalized Broyden — n = %d\n', n);
    fprintf('===========================================\n');

    %% Starting point suggested in [1]
    xbar = -ones(n,1);

    [f, gradf, hessf] = get_broyden_functions(p);


    %% ===== RUN 1: suggested starting point ==================
    fprintf('\n--- Starting point: xbar ---\n');

    try
        [xk,fk,gradfk_norm,k] = modified_newton_method2( ...
            xbar,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta);

        fprintf('||grad f|| = %.2e | iters = %d | f(x) = %.2e\n', ...
                gradfk_norm, k, fk);

    catch ME
        fprintf('FAILED (likely memory limit)\n');
        disp(ME.message);
    end

    %% ===== RUNS 2–6: random starting points =================
    for j = 1:5
        fprintf('\n--- Random start %d ---\n', j);

        x0 = xbar + 2*rand(n,1) - 1;

        try
            [xk,fk,gradfk_norm,k] = modified_newton_method2( ...
                x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta);

            fprintf('||grad f|| = %.2e | iters = %d | f(x) = %.2e\n', ...
                    gradfk_norm, k, fk);

        catch ME
            fprintf('FAILED (likely memory limit)\n');
            disp(ME.message);
        end
    end
end
