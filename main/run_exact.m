function run_exact()
% RUN_EXACT
% ------------------------------------------------------------
% Driver principale:
% - esegue Modified Newton (derivate esatte) su:
%     Problem 16 (Trig16) e Problem 31 (Broyden31)
% - per n = 2, 1e3, 1e4, 1e5
% - 6 starting points: xbar + 5 random in [xbar-1, xbar+1]
%
% Salva output in:
%   out_modified_exact/figures/
%   out_modified_exact/tables/

clear; clc; close all;

%% -------------------- Parametri richiesti --------------------
seed = 346710;                  % minima matricola del gruppo
rng(seed,"twister");

n_list = [2, 1e3, 1e4, 1e5];

% Parametri Modified Newton
kmax    = 1000;
tolgrad = 1e-6;
c1      = 1e-4;
rho     = 0.5;
btmax   = 60;
beta    = 1e-6;

%% -------------------- Cartelle output -------------------------
outdir = "out_modified_exact";
figdir = fullfile(outdir,"figures");
tabdir = fullfile(outdir,"tables");

if ~exist(outdir,"dir"), mkdir(outdir); end
if ~exist(figdir,"dir"), mkdir(figdir); end
if ~exist(tabdir,"dir"), mkdir(tabdir); end

%% -------------------- Problemi -------------------------------
probs = {@problem_trig16, @problem_broyden31};
pnames = ["Problem16_Trig", "Problem31_Broyden"];

% Tabella globale: accumulo righe in una cell array
allRows = {};

for p=1:numel(probs)

    [f,gradf,hessf,xbarfun] = probs{p}();
    pname = pnames(p);

    fprintf("\n==============================\n");
    fprintf("Eseguo %s\n", pname);
    fprintf("==============================\n");

    for n = n_list

        fprintf("\n--- n = %d ---\n", n);

        % 6 start: xbar + 5 random in [xbar-1, xbar+1]
        xbar = xbarfun(n);
        X0 = [xbar, xbar + (2*rand(n,5) - 1)];

        block = struct([]);

        for s=1:6
            store_seq = (n==2);

            tic;
            [xk,fk,gn,k,xseq,btseq,alphas,gnseq,fseq,tau_hist] = ...
                modified_newton_method(X0(:,s),f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta,store_seq);
            t = toc;

            success = (gn < tolgrad);
            rate_exp = exp_order_from_gradnorm(gnseq);
            if ~success, rate_exp = NaN; end

            fprintf("  start %d: it=%d | ||g||=%.2e | f=%.3e | t=%.2fs | succ=%d\n", ...
                s, k, gn, fk, t, success);

            % salvo nel blocco locale (per plots/tabelle)
            block(s).start = s;
            block(s).k = k;
            block(s).gn = gn;
            block(s).fk = fk;
            block(s).time = t;
            block(s).success = success;
            block(s).rate_exp = rate_exp;
            block(s).gnseq = gnseq;
            if store_seq
                block(s).xseq = xseq;
            end

            % salvo riga tabella globale
            startID = "xbar";
            if s>1, startID = "rand"+(s-1); end
            iters_over_max = sprintf("%d/%d", k, kmax);

            allRows(end+1,:) = {char(pname), n, char(startID), gn, iters_over_max, char(yesno(success)), rate_exp, t}; %#ok<AGROW>
        end

        % -------- Figure obbligatorie --------
        % Paths (solo n=2): contour + 6 percorsi
        if n == 2
            figP = plot_paths_2d(pname, f, X0, block);
            exportgraphics(figP, fullfile(figdir, sprintf("%s_n2_paths_exact.png", pname)), "Resolution", 300);
            close(figP);
        end

        % Rates: per ogni n (solo convergenti)
        figR = plot_rates(pname, n, block);
        exportgraphics(figR, fullfile(figdir, sprintf("%s_n%d_rates_exact.png", pname, n)), "Resolution", 300);
        close(figR);

        % Tabella stile Table 1 per (problema,n)
        Tn = make_table_exact(block, kmax);
        writetable(Tn, fullfile(tabdir, sprintf("%s_n%d_table_exact.csv", pname, n)));

    end
end

% Tabella globale (tutte le run)
Tall = cell2table(allRows, 'VariableNames', ...
    {'problem','n','start_pt_ID','grad_norm','iters_over_max','success_flag','rate_conv_exp','time_s'});

disp("Fatto. Output in: " + outdir);

end

%% =================== FUNZIONI DI SUPPORTO ===================

function y = yesno(flag)
if flag, y="yes"; else, y="no"; end
end

function p_est = exp_order_from_gradnorm(gnseq)
% Stima dell'ordine di convergenza p usando e_k = ||grad||
gn = gnseq(:);
gn = gn(isfinite(gn) & gn>0);
if numel(gn) < 6
    p_est = NaN; return;
end
tail = gn(max(1,end-10):end);
pvals = [];
for i=2:(numel(tail)-1)
    e1 = tail(i-1); e2 = tail(i); e3 = tail(i+1);
    if e1>e2 && e2>e3 && e1~=e2 && e2~=e3
        pk = log(e3/e2) / log(e2/e1);
        if isfinite(pk) && pk>0 && pk<5
            pvals(end+1) = pk; %#ok<AGROW>
        end
    end
end
if isempty(pvals), p_est = NaN; else, p_est = median(pvals); end
end

function T = make_table_exact(block, kmax)
% Tabella stile Table 1
m = numel(block);
startID = strings(m,1);
gradnorm = nan(m,1);
iters = strings(m,1);
succ = strings(m,1);
rate = nan(m,1);
time_s = nan(m,1);

for i=1:m
    s = block(i).start;
    if s==1, startID(i)="xbar"; else, startID(i)="rand"+(s-1); end
    gradnorm(i) = block(i).gn;
    iters(i) = sprintf("%d/%d", block(i).k, kmax);
    succ(i) = yesno(block(i).success);
    rate(i) = block(i).rate_exp;
    time_s(i) = block(i).time;
end
T = table(startID, gradnorm, iters, succ, rate, time_s, ...
    'VariableNames', {'start_pt_ID','grad_norm','iters_over_max','success_flag','rate_conv_exp','time_s'});


idx = (succ=="yes");
avgRow = {"Avg(successes)", mean(gradnorm(idx),'omitnan'), "", "-", mean(rate(idx),'omitnan'), mean(time_s(idx),'omitnan')};
T = [T; cell2table(avgRow, 'VariableNames', T.Properties.VariableNames)];

end

function fig = plot_rates(pname, n, block)
% Figura rates: log10(||grad||) vs k per tutte le run convergenti
fig = figure("Visible","off","Name",sprintf("%s n=%d rates",pname,n));
hold on; grid on;

plotted = false;
for i=1:numel(block)
    if block(i).success && ~isempty(block(i).gnseq)
        y = block(i).gnseq(:);
        plot(1:numel(y), log10(y + 1e-300), "LineWidth", 1);
        plotted = true;
    end
end

xlabel("iterazione k");
ylabel("log_{10}(||\nabla f(x_k)||)");
title(sprintf("%s (n=%d) - Rate sperimentali (solo convergenti)", pname, n));

if ~plotted
    text(0.5, 0.5, "Nessuna sequenza convergente per questo n", ...
        "Units","normalized","HorizontalAlignment","center");
end

hold off;
end

function fig = plot_paths_2d(pname, f, X0, block)
% Figura paths n=2: contour + percorsi
% -> 2 subplot: (1) vista completa, (2) zoom robusto (più leggibile)

% Raccolgo tutti i punti (start + sequenze) per stimare i limiti
P = X0';
for i=1:numel(block)
    if isfield(block(i),"xseq")
        xs = block(i).xseq';
        P = [P; xs]; %#ok<AGROW>
    end
end

% Limiti "full" basati su min/max
xminF = min(P(:,1)); xmaxF = max(P(:,1));
yminF = min(P(:,2)); ymaxF = max(P(:,2));

% Limiti "zoom" robusti (5%-95%) per evitare che 1 outlier schiacci tutto
qx = quantile(P(:,1), [0.05 0.95]);
qy = quantile(P(:,2), [0.05 0.95]);

padx = 0.2*(qx(2)-qx(1) + 1e-12);
pady = 0.2*(qy(2)-qy(1) + 1e-12);

xminZ = qx(1)-padx; xmaxZ = qx(2)+padx;
yminZ = qy(1)-pady; ymaxZ = qy(2)+pady;

% Funzione per disegnare contour in un box dato
    function draw_contour(ax, xmin, xmax, ymin, ymax)
        x1 = linspace(xmin, xmax, 220);
        x2 = linspace(ymin, ymax, 220);
        [X1,X2] = meshgrid(x1, x2);

        Z = zeros(size(X1));
        for ii=1:numel(X1)
            Z(ii) = f([X1(ii); X2(ii)]);
        end

        % Shift per evitare log di valori negativi e avere livelli informativi
        Zmin = min(Z(:));
        Zplot = log10((Z - Zmin) + 1e-12);

        contour(ax, X1, X2, Zplot, 30);
        grid(ax, "on");
        hold(ax, "on");
    end

fig = figure("Visible","off","Name",sprintf("%s n=2 paths",pname));

% --- subplot 1: vista completa ---
ax1 = subplot(1,2,1);
draw_contour(ax1, xminF, xmaxF, yminF, ymaxF);

for i=1:numel(block)
    if isfield(block(i),"xseq")
        xs = block(i).xseq;
        plot(ax1, xs(1,:), xs(2,:), "-o", "LineWidth", 1);
    end
end
plot(ax1, X0(1,:), X0(2,:), "kx", "MarkerSize", 8, "LineWidth", 1.5);
xlabel(ax1, "x_1"); ylabel(ax1, "x_2");
title(ax1, sprintf("%s (full)", pname));

% --- subplot 2: zoom leggibile ---
ax2 = subplot(1,2,2);
draw_contour(ax2, xminZ, xmaxZ, yminZ, ymaxZ);

for i=1:numel(block)
    if isfield(block(i),"xseq")
        xs = block(i).xseq;
        plot(ax2, xs(1,:), xs(2,:), "-o", "LineWidth", 1);
    end
end
plot(ax2, X0(1,:), X0(2,:), "kx", "MarkerSize", 8, "LineWidth", 1.5);
xlabel(ax2, "x_1"); ylabel(ax2, "x_2");
title(ax2, sprintf("%s (zoom)", pname));

sgtitle(sprintf("%s (n=2) - Contour + percorsi (exact)", pname));
end
