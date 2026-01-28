%% display_results: stampa tabelle e crea i grafici per un dato metodo
function display_results(res, method_name, dimensions, kmax, tolgrad)
    fprintf('\n--- Risultati: %s ---\n', method_name);

    % Per ogni dimensione costruisci e stampa tabella
    for i = 1:length(dimensions)
        n = dimensions(i);
        fprintf('\n--- Dimensione n = %d ---\n', n);
        fprintf('%-6s | %-10s | %-11s | %-11s | %-8s | %-8s | %-8s\n', 'start','grad.norm','iters/max','success','flag','rate(exp)','time(s)');
        fprintf(repmat('-',1,80)); fprintf('\n');

        n_success = 0; sum_gnorm = 0; sum_iters = 0; sum_time = 0; sum_rate = 0;
        for s = 1:6
            label = sprintf('n%d_pt%d', n, s);
            if ~isfield(res, label)
                % dato mancante
                fprintf('%-6s | %s\n', label, 'missing');
                continue;
            end
            r = res.(label);
            succ = pass_fail(r.gnorm, tolgrad);
            fprintf('%-6s | %-10.2e | %4d/%4d    | %-6s | %-8s | %-8.2f | %-8.2f\n', ...
                label, r.gnorm, r.iters, kmax, succ, r.flag, r.rate, r.time);

            if strcmp(succ,'yes')
                n_success = n_success + 1;
                sum_gnorm = sum_gnorm + r.gnorm;
                sum_iters = sum_iters + r.iters;
                sum_time = sum_time + r.time;
                if ~isnan(r.rate), sum_rate = sum_rate + r.rate; end
            end
        end

        if n_success > 0
            fprintf(repmat('-',1,80)); fprintf('\n');
            fprintf('%-6s | %-10.2e | %4d/%4d    | %-6s | %-8s | %-8.2f | %-8.2f\n', ...
                'Avg', sum_gnorm/n_success, round(sum_iters/n_success), kmax, '-', '-', sum_rate/n_success, sum_time/n_success);
        end
        fprintf(repmat('-',1,80)); fprintf('\n');
    end

    % --- Grafici: per l'ultima dimensione (ultima entry di dimensions) ---
    last_n = dimensions(end);
    std_label = sprintf('n%d_pt1', last_n);
    if isfield(res, std_label)
        figure('Name', [method_name ' - Convergence n=' num2str(last_n)]);
        subplot(2,1,1);
        gnorms_plot = generate_gnorm_seq(res.(std_label).xseq, @dummy_grad);
        semilogy(0:length(gnorms_plot)-1, gnorms_plot, '-o', 'LineWidth', 1.2, 'MarkerSize', 4);
        grid on;
        title([method_name ': Gradient Norm Decay (n=' num2str(last_n) ')']);
        xlabel('Iteration (k)'); ylabel('||\nabla f(x_k)||');

        % Execution time bar for the 6 starting points
        subplot(2,1,2);
        times = zeros(1,6);
        for s = 1:6
            label = sprintf('n%d_pt%d', last_n, s);
            if isfield(res, label)
                times(s) = res.(label).time;
            else
                times(s) = NaN;
            end
        end
        bar(times);
        set(gca, 'XTickLabel', {'x_bar','R1','R2','R3','R4','R5'});
        title([method_name ': Execution Time per Starting Point (n=' num2str(last_n) ')']);
        ylabel('Time (s)');
    end

    % --- Scalability: tempo medio e iterazioni medie (solo successi) ---
    avg_times = zeros(1,length(dimensions));
    avg_iters = zeros(1,length(dimensions));
    for i = 1:length(dimensions)
        n = dimensions(i);
        sum_t = 0; sum_it = 0; count = 0;
        for s = 1:6
            label = sprintf('n%d_pt%d', n, s);
            if isfield(res, label)
                r = res.(label);
                if r.gnorm < tolgrad
                    sum_t = sum_t + r.time;
                    sum_it = sum_it + r.iters;
                    count = count + 1;
                end
            end
        end
        if count>0
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
    grid on; title([method_name ': Average Time vs Dimension']);
    xlabel('Problem Dimension (n)'); ylabel('Average Time (s)');

    subplot(1,2,2);
    semilogx(dimensions, avg_iters, '-s', 'LineWidth', 2, 'MarkerSize', 8);
    grid on; title([method_name ': Average Iterations vs Dimension']);
    xlabel('Problem Dimension (n)'); ylabel('Average Newton Iterations');

end

function s = pass_fail(gnorm, tol)
    if gnorm < tol, s = 'yes'; else s = 'no'; end
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


