%% ========================================================================
%  RUNNER: 6 starting points + SUCCESS (adattato per outputs)
% ========================================================================
function res_fd = run_6starts_success(store, X0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta)
% Esegue Modified Newton su n punti iniziali.
% Se store==true salva tutte le iterazioni in xseq per ogni start.
% Ritorna array di struct con xseq per ogni start.

num_pts = size(X0,2);

% Preallocazione dell'array di struct
res_fd(num_pts) = struct('xseq',[],'fk',[],'gn',[],'iters',[],'time',[],'success',[]);

for s = 1:num_pts
    x0 = X0(:,s);

    tic;
    try
        % truncated_newton_method deve restituire xseq completo se store==true
        [~, fk, gn, it, xseq] = truncated_newton_method(x0, f, gradf, hessf, ...
                                    kmax, tolgrad, c1, rho, btmax, beta);
    catch
        fk = NaN; gn = Inf; it = 0;   % fallback in caso di errore
    end
    t = toc;

    succ = (gn < tolgrad);

    % Salva i dati nello struct corrispondente allo start s
    res_fd(s).fk      = fk;
    res_fd(s).gn      = gn;
    res_fd(s).iters   = it;
    res_fd(s).time    = t;
    res_fd(s).success = succ;

    if store
        res_fd(s).xseq = xseq;  % salva sequenza iterativa completa
    else
        res_fd(s).xseq = x0;     % almeno punto iniziale
    end

    % Stampa informazioni sul run
    startID = "xbar";
    if s > 1, startID = "rand" + (s-1); end
    fprintf("%s: it=%d | ||g||=%.2e | f=%.3e | t=%.2fs | succ=%d\n", ...
            startID, it, gn, fk, t, succ);
end
end
