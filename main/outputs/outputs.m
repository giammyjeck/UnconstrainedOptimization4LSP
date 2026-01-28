function outputs(results, f, gradf, pname)
% OUTPUTS aggiornata per struct results{truncated, modified}
% genera grafici, tabelle e traiettorie per ogni metodo

methods = fieldnames(results);  % truncated, modified
outdir = "C:/Users/Utente/Desktop/Corsi/Numerical optimization for large scale problems and Stochastic Optimization/NumericalO4LSP/main/outputs/";
figdir = fullfile(outdir,"figures");
tabdir = fullfile(outdir,"tables");
if ~exist(outdir,"dir"), mkdir(outdir); end
if ~exist(figdir,"dir"), mkdir(figdir); end
if ~exist(tabdir,"dir"), mkdir(tabdir); end

kmax = 1000; tolgrad = 1e-6;

for m = 1:length(methods)
    method = methods{m};
    res_method = results.(method);

    % Raccogli dimensioni dai label: assumo n%d_pt%d
    labels = fieldnames(res_method);
    dims = unique(cellfun(@(s) sscanf(s,'n%d_pt'), labels));
    
    allRows = {};  % globale
    
    for i = 1:length(dims)
        n = dims(i);
        % estrai tutti i punti di partenza per questa dimensione
        run_labels = labels(contains(labels,sprintf('n%d_',n)));
        num_runs = length(run_labels);
        tableRows = cell(num_runs,6);
        success_indices = [];

        figR = figure('Visible','off','Name', sprintf('%s Rates n=%d', method, n));
        hold on; grid on;

        for s = 1:num_runs
            r = res_method.(run_labels{s});
            xseq = r.xseq; gnorms = zeros(1,size(xseq,2));
            for it=1:size(xseq,2), gnorms(it) = norm(gradf(xseq(:,it))); end

            % Calcolo rate sperimentale usando estimate_rate
            rate_exp = estimate_rate(xseq, gradf);
            success = r.gnorm < tolgrad;

            startID = "R"+(s-1); if s==1, startID="x_bar"; end
            iters_str = sprintf("%d/%d", r.iters, kmax);
            succ_str = "no"; if success, succ_str="yes"; end
            tableRows(s,:) = {char(startID), r.gnorm, iters_str, char(succ_str), rate_exp, r.time};
            allRows(end+1,:) = {char(pname), method, n, char(startID), r.gnorm, iters_str, char(succ_str), rate_exp, r.time};

            if success
                plot(0:length(gnorms)-1, log10(gnorms+1e-20),'LineWidth',1.2);
                success_indices = [success_indices,s]; %#ok<AGROW>
            end
        end

        xlabel('Iteration k'); ylabel('log_{10}(||\nabla f(x_k)||)');
        title(sprintf('%s (n=%d) - Convergence Rates', method, n));
        exportgraphics(figR, fullfile(figdir,sprintf('%s_%s_n%d_rates.png', pname, method, n)));
        close(figR);

        % Tabella CSV
        Tn = cell2table(tableRows, 'VariableNames', {'start_pt_ID','grad_norm','iters_over_max','success_flag','rate_conv_exp','time_s'});
        if ~isempty(success_indices)
            avg_gn = mean([tableRows{success_indices,2}]);
            avg_rate = mean([tableRows{success_indices,5}],'omitnan');
            avg_time = mean([tableRows{success_indices,6}]);
            avgRow = {"Avg(successes)", avg_gn, "", "-", avg_rate, avg_time};
            Tn = [Tn; cell2table(avgRow,'VariableNames',Tn.Properties.VariableNames)];
        end
        writetable(Tn, fullfile(tabdir,sprintf('%s_%s_n%d_table.csv', pname, method, n)));
    end

    % Scalability
    avg_times = zeros(1,length(dims)); avg_iters = zeros(1,length(dims));
    for i=1:length(dims)
        n=dims(i); run_labels = labels(contains(labels,sprintf('n%d_',n)));
        times=[]; iters=[]; 
        for s=1:length(run_labels)
            r=res_method.(run_labels{s});
            if r.gnorm < tolgrad
                times=[times,r.time]; iters=[iters,r.iters]; %#ok<AGROW>
            end
        end
        avg_times(i)=mean(times); avg_iters(i)=mean(iters);
    end

    figS = figure('Name',[method ' Scalability'],'Color','w','Visible','on');
    subplot(1,2,1); loglog(dims,avg_times,'-o','LineWidth',2); grid on; xlabel('n'); ylabel('Avg Time (s)'); title('Scalabilità Tempo');
    subplot(1,2,2); semilogx(dims,avg_iters,'-s','LineWidth',2); grid on; xlabel('n'); ylabel('Avg Iters'); title('Robustezza Iterazioni');
    exportgraphics(figS, fullfile(figdir,sprintf('%s_%s_scalability.png', pname, method)));

end

Tall = cell2table(allRows, 'VariableNames', {'problem','method','n','start_pt_ID','grad_norm','iters_over_max','success_flag','rate_conv_exp','time_s'});
writetable(Tall, fullfile(tabdir, sprintf('%s_GLOBAL_table.csv', pname)));

end


function p = estimate_rate(xseq, gradf)
    % Stima esponente p basata sulle ultime tre norme di gradiente
    if isempty(xseq) || size(xseq,2) < 4
        p = NaN; return;
    end
    g3 = norm(gradf(xseq(:, end)));
    g2 = norm(gradf(xseq(:, end-1)));
    g1 = norm(gradf(xseq(:, end-2)));
    if any([g1,g2,g3] < 1e-14)
        p = NaN; return;
    end
    p = log(g3/g2) / log(g2/g1);
    if p < 0 || p > 3, p = NaN; end
end
