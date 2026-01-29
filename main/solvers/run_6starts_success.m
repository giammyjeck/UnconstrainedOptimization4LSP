%% ========================================================================
%  RUNNER: 6 starting points + SUCCESS
% ========================================================================
function res = run_6starts_success(store, X0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta)
% Esegue Modified Newton su 6 punti iniziali.
% Ritorna struct con:
%   - iters, ||g|| finale, f finale, time, success boolean

num_pts = size(X0,2);

res = struct();
res.success = false(num_pts,1);
res.iters   = zeros(num_pts,1);
res.gn      = zeros(num_pts,1);
res.fk      = zeros(num_pts,1);
res.time    = zeros(num_pts,1);

for s = 1:num_pts
    x0 = X0(:,s);

    tic;
    try
        [~,fk, gn, it] = modified_newton_method(x0, f, gradf, hessf, ...
            kmax, tolgrad, c1, rho, btmax, beta);
        %[xk,fk,gradfk_norm,k,xseq,btseq,alphas,gradfk_seq,fk_seq,tau_new,pks]
    catch
        % se qualcosa va storto, marchiamo fail
        fk = NaN; gn = Inf; it = 0;
    end
    t = toc;

    succ = (gn < tolgrad);

    res.fk(s)      = fk;
    res.gn(s)      = gn;
    res.iters(s)   = it;
    res.time(s)    = t;
    res.success(s) = succ;

    startID = "xbar";
    if s > 1, startID = "rand" + (s-1); end

    fprintf("    %s: it=%d | ||g||=%.2e | f=%.3e | t=%.2fs | succ=%d\n", ...
        startID, it, gn, fk, t, succ);
end
end