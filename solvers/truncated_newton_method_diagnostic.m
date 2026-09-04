function [xk,fk,gradfk_norm,k,xseq,btseq,pks,inner_iters,trace] = truncated_newton_method_diagnostic(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,max_cg)

% TRUNCATED_NEWTON_METHOD_DIAGNOSTIC - come truncated_newton_method, ma
% registra OGNI passo interno (CG e backtracking) per poter verificare
% visivamente/step-by-step se le scelte fatte dall'algoritmo sono
% corrette.
%
% trace(k) ha i campi "riassuntivi" gia' visti in
% truncated_newton_method_verbose.m (x_before, eta, cg_iters, cg_exit,
% pk, pk_norm, alpha0, alpha_final, bt_steps, x_after, fk, fk_new,
% gradnorm), PIU':
%
%   trace(k).cg_trace(j)  : per ogni iterazione CG j = 1..cg_iters
%       .z          : iterata parziale z_j (candidato di direzione dopo
%                     j passi CG; z_0 = 0 non e' salvato)
%       .dk         : direzione coniugata usata al passo j
%       .curv       : curvatura dk'*Hk*dk misurata al passo j
%       .curv_thresh: soglia di curvatura usata per il test
%       .decision   : 'accepted' (curvatura positiva, passo CG eseguito)
%                     | 'stopped_curvature' (curvatura troppo bassa/negativa)
%                     | 'stopped_tol' (||r|| <= eta raggiunta dopo il passo)
%
%   trace(k).bt_trace(t)  : per ogni tentativo di backtracking t = 1..
%       .alpha       : alpha provato
%       .x_trial     : x_before + alpha*pk
%       .f_trial     : f(x_trial)
%       .armijo_rhs  : f(x_before) + c1*alpha*(grad'*pk)  (soglia Armijo)
%       .accepted    : true se f_trial <= armijo_rhs (passo accettato)
%
% INPUT/OUTPUT principali: identici a truncated_newton_method.m

if ~isnumeric(x0) || ~isvector(x0)
    error('x0 must be a numeric column vector.');
end
xk = x0(:);
n = length(xk);

fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);

farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;

xseq        = zeros(n, kmax+1);
btseq       = zeros(1, kmax);
pks         = zeros(n, kmax);
inner_iters = zeros(1, kmax);

trace = struct('x_before',{}, 'eta',{}, 'cg_iters',{}, 'cg_exit',{}, ...
                'pk',{}, 'pk_norm',{}, 'alpha0',{}, 'alpha_final',{}, ...
                'bt_steps',{}, 'x_after',{}, 'fk',{}, 'fk_new',{}, ...
                'gradnorm',{}, 'cg_trace',{}, 'bt_trace',{});

k = 0;
xseq(:,1) = xk;
while k < kmax && gradfk_norm >= tolgrad

    eta_k = min(0.5, sqrt(gradfk_norm))*gradfk_norm;

    z = zeros(n,1);
    Hk = hessf(xk);
    rk = gradfk;
    dk = -rk;
    r_old = rk'*rk;

    j = 0;
    cg_exit = 'maxcg';
    p_tn = -gradfk; % fallback
    cg_trace = struct('z',{}, 'dk',{}, 'curv',{}, 'curv_thresh',{}, 'decision',{});

    while j < max_cg
        Hdk = Hk*dk;
        curv = dk'*Hdk;
        curv_thresh = 1e-10 * norm(dk)^2;

        if curv <= curv_thresh
            if j == 0
                p_tn = -gradfk;
            else
                p_tn = z;
            end
            cg_exit = 'curvature';
            cg_trace(end+1) = struct('z', z, 'dk', dk, 'curv', curv, ...
                'curv_thresh', curv_thresh, 'decision', 'stopped_curvature'); %#ok<AGROW>
            break
        else
            alpha_j = r_old/curv;
            z = z + alpha_j * dk;
            rk = rk + alpha_j * Hdk;

            if norm(rk) <= eta_k
                p_tn = z;
                cg_exit = 'tol';
                cg_trace(end+1) = struct('z', z, 'dk', dk, 'curv', curv, ...
                    'curv_thresh', curv_thresh, 'decision', 'stopped_tol'); %#ok<AGROW>
                break;
            end

            cg_trace(end+1) = struct('z', z, 'dk', dk, 'curv', curv, ...
                'curv_thresh', curv_thresh, 'decision', 'accepted'); %#ok<AGROW>

            rk_new = rk'*rk;
            beta_j = rk_new/r_old;
            dk = -rk + beta_j * dk;
            r_old = rk_new;
            j = j+1;
        end
    end

    j_cg = j;
    if j == max_cg
        p_tn = z;
        cg_exit = 'maxcg';
    end
    pk = p_tn;

    if norm(pk) <= 1e-12 && gradfk_norm < tolgrad
        xseq = xseq(:,1:k+1);
        btseq = btseq(1:k); pks = pks(:,1:k); inner_iters = inner_iters(1:k);
        return;
    end

    % --- BACKTRACKING (log di ogni alpha provato) ---
    x_before = xk;
    fk_before = fk;
    gradnorm_before = gradfk_norm;

    alpha = 1;
    alpha0 = alpha;
    c1_gradfk_pk = c1 * gradfk' * pk;

    bt_trace = struct('alpha',{}, 'x_trial',{}, 'f_trial',{}, 'armijo_rhs',{}, 'accepted',{});

    xnew = xk + alpha * pk;
    fnew = f(xnew);
    armijo_rhs = farmijo(fk, alpha, c1_gradfk_pk);
    bt_trace(end+1) = struct('alpha', alpha, 'x_trial', xnew, 'f_trial', fnew, ...
        'armijo_rhs', armijo_rhs, 'accepted', fnew <= armijo_rhs);

    bt = 0;
    while bt < btmax && fnew > armijo_rhs
        alpha = rho * alpha;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        armijo_rhs = farmijo(fk, alpha, c1_gradfk_pk);
        bt = bt + 1;
        bt_trace(end+1) = struct('alpha', alpha, 'x_trial', xnew, 'f_trial', fnew, ...
            'armijo_rhs', armijo_rhs, 'accepted', fnew <= armijo_rhs); %#ok<AGROW>
    end

    stalled = (bt == btmax && fnew > armijo_rhs);

    k = k+1;
    trace(k).x_before    = x_before;
    trace(k).eta         = eta_k;
    trace(k).cg_iters    = j_cg;
    trace(k).cg_exit     = cg_exit;
    trace(k).pk          = pk;
    trace(k).pk_norm     = norm(pk);
    trace(k).alpha0      = alpha0;
    trace(k).alpha_final = alpha;
    trace(k).bt_steps    = bt;
    trace(k).fk          = fk_before;
    trace(k).gradnorm    = gradnorm_before;
    trace(k).cg_trace    = cg_trace;
    trace(k).bt_trace    = bt_trace;

    if stalled
        trace(k).x_after = x_before;
        trace(k).fk_new  = fk_before;
        xseq(:, k+1)   = xk;
        btseq(k)       = bt;
        pks(:, k)      = pk;
        inner_iters(k) = j_cg;
        break;
    end

    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);

    trace(k).x_after = xk;
    trace(k).fk_new  = fk;

    xseq(:, k+1)   = xk;
    btseq(k)       = bt;
    pks(:, k)      = pk;
    inner_iters(k) = j_cg;

end

xseq        = xseq(:, 1:k+1);
btseq       = btseq(1:k);
pks         = pks(:, 1:k);
inner_iters = inner_iters(1:k);

end
