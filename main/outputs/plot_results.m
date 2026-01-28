function plot_results(results, gradf, dimensions, tolgrad, method_name)
% plot_results: genera grafici di convergenza e scalabilità per un dato metodo
% results: struttura contenente i risultati per ciascun punto di partenza
% gradf: handle del gradiente
% dimensions: vettore delle dimensioni dei test
% tolgrad: tolleranza gradiente
% method_name: nome metodo (stringa) per titoli e nomi figure

%% Convergence & Execution Time (last dimension)
last_n = dimensions(1);
std_label = sprintf('n%d_pt1', last_n);

if isfield(results, std_label)
    figure('Name', [method_name ' - Convergence n=' num2str(last_n)]);
    
    % Gradient norm decay
    subplot(2,1,1);
    gnorms_plot = generate_gnorm_seq(results.(std_label).xseq, gradf);
    semilogy(0:length(gnorms_plot)-1, gnorms_plot, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    grid on;
    title([method_name ': Gradient Norm Decay (n=' num2str(last_n) ')']);
    xlabel('Iteration (k)'); ylabel('||\nabla f(x_k)||');
    
    % Execution time per starting point
    subplot(2,1,2);
    times = zeros(1,6);
    for s = 1:6
        label = sprintf('n%d_pt%d', last_n, s);
        if isfield(results, label)
            times(s) = results.(label).time;
        else
            times(s) = NaN;
        end
    end
    bar(times, 'FaceColor', [0.3 0.5 0.9]);
    set(gca, 'XTickLabel', {'x_bar','R1','R2','R3','R4','R5'});
    title([method_name ': Execution Time per Starting Point (n=' num2str(last_n) ')']);
    ylabel('Time (s)');
end

%% Scalability analysis (average time & iterations)
avg_times = zeros(1,length(dimensions));
avg_iters = zeros(1,length(dimensions));

for i = 1:length(dimensions)
    n = dimensions(i);
    sum_t = 0; sum_it = 0; count = 0;
    for s = 1:6
        label = sprintf('n%d_pt%d', n, s);
        if isfield(results, label)
            r = results.(label);
            if r.gnorm < tolgrad
                sum_t = sum_t + r.time;
                sum_it = sum_it + r.iters;
                count = count + 1;
            end
        end
    end
    if count > 0
        avg_times(i) = sum_t / count;
        avg_iters(i) = sum_it / count;
    else
        avg_times(i) = NaN;
        avg_iters(i) = NaN;
    end
end

figure('Name',[method_name ' - Scalability']);

subplot(1,2,1);
loglog(dimensions, avg_times, '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title([method_name ': Average Time vs Dimension']);
xlabel('Problem Dimension (n)'); ylabel('Average Time (s)');

subplot(1,2,2);
semilogx(dimensions, avg_iters, '-s', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title([method_name ': Average Iterations vs Dimension']);
xlabel('Problem Dimension (n)'); ylabel('Average Newton Iterations');
end


%% --- Funzioni di supporto per stime / sequenze ---

function gnorms = generate_gnorm_seq(xseq, gradf)
    if isempty(xseq)
        gnorms = [];
        return;
    end
    num_pts = size(xseq, 2);
    gnorms = zeros(1, num_pts);
    for j = 1:num_pts
        gnorms(j) = norm(gradf(xseq(:,j)));
    end
end


