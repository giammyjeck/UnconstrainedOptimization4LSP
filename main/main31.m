%% Main Script for Problem 16
clear; clc; close all;

seed = 346710; 
rng(seed);
addpath(genpath(pwd))


% Defining the problem
[f, gradf, hessf, xbar_gen] = problem_broyden31();

% Parameters definition
dimensions = [2, 10^3, 10^4, 10^5]; 
kmax = 1000;
tolgrad = 1e-6;
c1 = 1e-4;      % Standard Armijo parameter
rho = 0.5;      % Backtracking contraction factor
btmax = 20;
max_cg = 500;   % Max inner iterations for the conjugate gradient solving method in the truncated one

% Data structures to store results 
results_tn = struct();

for i = 1:length(dimensions) % Loop on the problem dimension
    n = dimensions(i);
    fprintf('\n--- Test dimension n = %d (Broyden 31) ---\n', n);

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
    
    for s = 1:6 % Loop on the 6 starting point
        x0_curr = all_x0(:, s);
        point_label = sprintf('n%d_pt%d', n, s);
        
        tic;
        [xk_tn, fk_tn, gnorm_tn, k_tn, xseq_tn, btseq_tn, pks_tn, inner_tn] = ...
            truncated_newton_method(x0_curr, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, max_cg);
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
        %Salvataggio
        results_tn.(point_label).time = time_tn;
        results_tn.(point_label).iters = k_tn;
        results_tn.(point_label).gnorm = gnorm_tn;

        if s == 1
            % Salviamo xseq solo per il punto standard per poterlo plottare dopo
            results_tn.(point_label).xseq = xseq_tn;
        end

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

%% --- VISUALIZATION (Last n) ---
% Il primo grafico rappresenta il profilo di convergenza del metodo Truncated Newton 
% attraverso l'andamento della norma del gradiente in scala semilogaritmica. 
% La pendenza della curva permette di valutare visivamente il tasso di convergenza: 
% una traiettoria che accelera verso il basso indica una convergenza superlineare, 
% obiettivo principale del metodo Newton-CG ottenuto grazie alla condizione di 
% forcing eta_k. Questo grafico è fondamentale per verificare il raggiungimento 
% della tolleranza impostata (tolgrad) e per validare numericamente l'efficacia 
% delle direzioni di discesa calcolate dal gradiente coniugato interno.

% Il secondo grafico analizza l'efficienza computazionale e la robustezza del 
% solutore al variare del punto di partenza. Il confronto tra il tempo di calcolo 
% del punto standard (x_bar) e dei punti generati casualmente (R1-R5) evidenzia 
% la sensibilità dell'algoritmo alla geometria della funzione obiettivo nelle 
% diverse regioni dello spazio di ricerca. Per problemi di grandi dimensioni come 
% il presente, questo istogramma permette di confermare la scalabilità del metodo 
% Truncated Newton, il quale, evitando il calcolo esatto dell'inversa dell'Hessiana, 
% mantiene tempi di esecuzione contenuti anche per n elevati, garantendo al contempo 
% stabilità numerica indipendentemente dall'inizializzazione.



last_n = dimensions(end);
std_label = sprintf('n%d_pt1', last_n);
if isfield(results_tn, std_label)
    figure('Name', ['Convergence Analysis n = ' num2str(last_n)]);
    
    subplot(2,1,1);
    gnorms_plot = generate_gnorm_seq(results_tn.(std_label).xseq, gradf);
    semilogy(0:results_tn.(std_label).iters, gnorms_plot, '-bo', 'LineWidth', 1.2, 'MarkerSize', 3);
    grid on;
    title(['Gradient Norm Decay (n=' num2str(last_n) ')']);
    xlabel('Iteration (k)'); ylabel('||\nabla f(x_k)||');
    
    subplot(2,1,2);
    all_times = arrayfun(@(s) results_tn.(sprintf('n%d_pt%d', last_n, s)).time, 1:6);
    bar(all_times, 'FaceColor', [0.3 0.5 0.9]);
    set(gca, 'XTickLabel', {'x_bar', 'R1', 'R2', 'R3', 'R4', 'R5'});
    title('Execution Time per Starting Point');
    ylabel('Time (s)');
end

%% --- GLOBAL SCALABILITY ANALYSIS ---
%% Analisi della Scalabilità al variare della Dimensione (n)
% I grafici di scalabilità permettono di valutare l'efficienza del metodo Truncated Newton 
% nel gestire problemi di dimensioni crescenti. Il primo grafico (Log-Log) mette in 
% relazione il tempo medio di esecuzione con la dimensione n: un andamento prossimo 
% alla linearità conferma che l'algoritmo scala in modo efficiente, evitando l'onere 
% computazionale cubico tipico del Newton puro grazie all'approssimazione del sistema 
% lineare tramite il Gradiente Coniugato. Il secondo grafico (Semilog-X) mostra invece 
% come il numero di iterazioni esterne rimanga pressoché costante o cresca in modo 
% estremamente contenuto al variare di n; questa proprietà, nota come "indipendenza 
% dalla dimensione", è un indicatore di robustezza algoritmica e dimostra come 
% l'informazione del secondo ordine (Hessiana) venga sfruttata efficacemente per 
% mantenere un'elevata velocità di convergenza anche su problemi a larga scala.
figure('Name', 'Scalability Analysis: Truncated Newton');

avg_times = zeros(1, length(dimensions));
avg_iters = zeros(1, length(dimensions));

for i = 1:length(dimensions)
    n = dimensions(i);
    sum_t = 0; sum_it = 0; count = 0;
    for s = 1:6
        label = sprintf('n%d_pt%d', n, s);
        if results_tn.(label).gnorm < tolgrad
            sum_t = sum_t + results_tn.(label).time;
            sum_it = sum_it + results_tn.(label).iters;
            count = count + 1;
        end
    end
    if count > 0
        avg_times(i) = sum_t / count;
        avg_iters(i) = sum_it / count;
    end
end

subplot(1,2,1);
loglog(dimensions, avg_times, '-ro', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title('Computational Effort vs Dimension');
xlabel('Problem Dimension (n)'); ylabel('Average Time (s)');

subplot(1,2,2);
semilogx(dimensions, avg_iters, '-bs', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title('Iterations vs Dimension');
xlabel('Problem Dimension (n)'); ylabel('Average Newton Iterations');

%% --- HELPER FUNCTIONS ---
function p = estimate_rate(xseq, gradf)
    if size(xseq, 2) < 4, p = NaN; return; end
    g3 = norm(gradf(xseq(:, end)));
    g2 = norm(gradf(xseq(:, end-1)));
    g1 = norm(gradf(xseq(:, end-2)));
    if g1 < 1e-14 || g2 < 1e-14 || g3 < 1e-14, p = NaN; return; end
    p = log(g3/g2) / log(g2/g1);
    if p < 0 || p > 3, p = NaN; end
end

function gnorms = generate_gnorm_seq(xseq, gradf)
    num_pts = size(xseq, 2); 
    gnorms = zeros(1, num_pts);
    for j = 1:num_pts
        gnorms(j) = norm(gradf(xseq(:,j)));
    end
end