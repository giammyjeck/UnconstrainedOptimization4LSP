% ============================================================
% Problem 5: Generalized Broyden Tridiagonal Function
% ============================================================

clear; clc; close all;

% 1. Inizialitation phase
% Random seed
student_id = 346710; % Francesca's ID
rng(student_id);

p = 7/3; 

kmax = 1000; % Maximum number of iterations
tolgrad = 1e-6; % Gradient tolerance
c1 = 1e-4;   % Armijo value
rho = 0.5;   % Reduction parameter for backtracking
btmax = 20;  % Maximum number for halving alpha
beta = 1e-3; % Correction parameter for the Hessian


%% SINGLE EXPERIMENT RUN
dim = 2;
[f, grad, hess] = get_broyden_functions(p);
xStarting = -ones(dim,1);

fprintf('\n--- Starting point: xStarting ---\n');
try
    tic;
    [xk,fk,gradfk_norm,k] = modified_newton_method( ...
        xStarting,f,grad,hess,kmax,tolgrad,c1,rho,btmax,beta);
    time = toc;
    
    fprintf('Iters: %d, Final f: %.2e, GradNorm: %.2e, Time: %.4fs\n', ...
         k, fk, gradfk_norm, time);

catch ME
    fprintf('FAILED (likely memory limit)\n');
    disp(ME.message);
end


%% 2. FULL EXPERIMENT RUN for actual configuration 
disp('***   PROBLEM NUMBER 5   ***');
dim = [2, 1e3, 1e4, 1e5];
experimentalMAtrix
for n = dim

    fprintf('\n--- Dimension n = %d ---\n', n);
    
    % Starting point
    x_bar = -ones(n, 1);
    
    % Vector containing the starting points: x_bar + 5 random ones
    X0 = [x_bar, x_bar - 1 + 2 * rand(n, 5)];

    % Calling the function which computes the evaluation of the broyden
    % function, its gradient and the hessian matrix
    [f, grad, hess] = get_broyden_functions(p);
    
    % Loop for applying the MNM with different starting point
    for j = 1:size(X0, 2)
        x0 = X0(:, j);
        
        tic;
        [xk,fk,gradfk_norm,k,xseq,btseq,tau_new,alphas,pks] = modified_newton_method(...
            x0, f, grad, hess, kmax, tolgrad, c1, rho, btmax, beta);
        time = toc;
        
        fprintf('Start Point %d: Iters: %d, Final f: %.2e, GradNorm: %.2e, Time: %.4fs\n', ...
            j, k, fk, gradfk_norm, time);
    end
end


% --- NOTE PER L'IMPLEMENTAZIONE ---
% Per gestire n = 1e5, è obbligatorio usare matrici SPARSE (comando 'spdiags') 
% per l'Hessiana, altrimenti la memoria RAM si esaurirà immediatamente.
% La struttura tridiagonale della funzione di Broyden permette al metodo di 
% Newton di risolvere il sistema Bk*pk = -gradfk in tempo lineare O(n).
% I parametri scelti (c1=1e-4, rho=0.5) sono standard per garantire la 
% convergenza globale tramite backtracking di Armijo. 
% Il parametro beta=1e-3 assicura che Bk sia ben condizionata, 
% evitando errori numerici nel solutore diretto R\(R'\(-gradfk)).