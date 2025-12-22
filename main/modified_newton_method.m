function [xk,fk,gradfk_norm,k,xseq,btseq] = modified_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

% MODIFIED_NEWTON_METHOD  Modified Newton method with Hessian correction
% [xk, fk, gradfk_norm, k, xseq, btseq, tau_new] = ...
%     modified_newton_method(x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta)
%
%   This functions is aimed to solve large scale numerical optimization
%   problems using the modified Newton method with backtracking techniques.
%
% INPUT:
%   x0      : initial point (column vector)
%   f       : function handle for the scalar objective function f(x)
%   gradf   : function handle for the gradient gradf(x)
%   hessf   : function handle for the Hessian hessf(x)
%   kmax    : maximum number of iterations
%   tolgrad : tolerance on the gradient norm for the stopping condition
%   c1, rho, btmax : parameters for Armijo/backtracking (0 < c1 < 1, 0 < rho < 1)
%   beta    : minimum initial increment for the Hessian correction (beta > 0)
%
% OUTPUT:
%   xk, fk, gradfk_norm : final point, objective value, gradient norm at the solution
%   k        : number of iterations performed
%   xseq     : sequence of iterates [x0, x1, ..., xk]  (n × (k+1))
%   btseq    : number of backtracking steps at each iteration (1 × k)
%   tau_new  : (maxit+1) × k matrix storing the tau values used at each iteration



% Function handle for the armijo condition
farmijo = @(fk, alpha, c1_gradfk_pk) ...
    fk + alpha * c1_gradfk_pk;


% Initializations
xseq = zeros(length(x0), kmax); % matrix to store computed solution 
btseq = zeros(1, kmax);         % vector to store number of backtracking iteration  

% !!! i parametri sotto dovrebbero essere settati con coerenza
% quindi controllare e definire la coerenza
tol = 1e-6;                     % tol to check the positivness of Hk diagonal
maxit = 100;                    % max number of iteration to check the positivness of Hk diagonal

xk = x0;
fk = f(xk);
k = 0;
gradfk = gradf(xk);
gradfk_norm = norm(gradfk); % !!! salvo la norma per ragioni computazionali, verificare dove altrimenti dovrei ricalcolarla
tau_new = zeros(maxit+1,kmax); % matrix to save the values of tau_k

% !!! dovremmo controllare che già la gradfk_norm iniziale non sia <
% tolgrad altrimenti neanche parto, sollevare un flag
while k < kmax && gradfk_norm >= tolgrad
    
    % Compute the hessian matrix
    Hk = hessf(xk);
    
    % Check if the hessian matrix is positive definite
    % !!! Questa è una condizione che non garantisce la definita
    % positività, è condizione necessaria ma non sufficiente!
    % eventualmente provare altre condizioni
    diagHk = diag(Hk);
    isPositive = all(diagHk>tol);

    if isPositive == true
        tau_k = 0; % no need to add a correctional term 
    else
        tau_k = beta-min(diagHk); %check se va messo max(0,)
        % if min(diagHK) is negative ? 

    end
     
    %Bk = Hk+tau_k*eye(size(Hk));

    % Once the parameter tau is found then the correction of the hessian 
    % can be built.
    % In order to check if the corrected matrix is positive definite, we 
    % attempt to perform the incomplete choleski factorization: 
    % if Bk is not positive definite, then you get an error.

    tauk = zeros(maxit+1,1); % vector to store the history of correction term per iteration
    tauk(1)= tau_k;

    for j = 1:maxit
            Bk = Hk+tauk(j)*speye(size(Hk)); % dovremmo controllare che Bk sia simmetrica prima di usare chol? 
           
            [R,flag] = chol(Bk);
            % Chech if the correction is good enough (Is Bk positive definite?)

            if flag == 0
                disp('Bk is positive definite')
                break
            elseif flag == 1
        
                % If the Bk in not positive definite, then we need to correct it.
                % new value of tau_k !!! perché proprio questo valore?
                tauk(j+1) = max(beta, 2*tauk(j)); % controllare che tau non possa essere negativo?
            else
                disp('Error')
            end
    end

    % Once we obtain a matrix Bk which is positive definite we can solve the
    % following system with pcg method in order to find the 
    % descent direction. !!! INSTEAD we use a direct solver 
     
    %pk = pcg(Bk, -gradfk, [], [], R, R');
    pk = R\(R'\(-gradfk));
    
    % NOTE: there's no need to check if pk is a descent direction because
    % of remark2 !!! CONTROLLARE SU TEORIA PERCHEé
    
    
    % Reset the value of alpha
    alpha = 1; % this is the parameter used for linesearch
    % !!! for teoretical properties you need alpha to be 1
    
    % Compute the candidate new xk
    xnew = xk + alpha * pk;
    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;

    % Backtracking strategy from here...: 
    % 2nd condition is the Armijo condition not satisfied
    
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        % Reduce the value of alpha
        alpha = rho * alpha;
        % Update xnew and fnew w.r.t. the reduced alpha
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
    end

    % Check if the maximum number of backtracking iterations is reached
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        disp('Maximum backtracking iterations reached, stopping.');
        break;
    end

    % ...to here
    
    % Update xk, fk, gradfk_norm
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    

    % Increase the step by one
    k = k + 1; % !!! CONTROLLARE se è  più robusto incrementare dopo aver salvato

    % Store current xk in xseq
    xseq(:, k) = xk;
    % Store bt iterations in btseq
    btseq(k) = bt;
    %Store current tauk values
    tau_new(:, k) = tauk;


end


% "Cut" xseq and btseq to the correct size
xseq = xseq(:, 1:k);
btseq = btseq(1:k);
% "Add" x0 at the beginning of xseq (otherwise the first el. is x1)
xseq = [x0, xseq];


end


% TO DO:

% INPUT: 
% controllare e scrivere la dimensione e il tipo expected di ogni parametro
% aggiungere dei controlli sugli input


%BACKTRACKING
%
% rho dalla teoria del backtracking sappiamo che può essere fisso o chosen
% by interpolation rho \in [\rho_l, \rho_u]
%
% Controllare quale valore alpha soddisfa armijo perché se alpha <<1
% abbiamo stagnation
%
% ATTENZIONE: Non fare l'errore di permettere che alpha sia < eps, bisogna
% stare attenti a scegliere un rho e btmax che non permetta alpha<eps,
% NEPPURE TEORICAMENTE, cioè quando fissiamo i parametri cerchiamo delle
% condizioni che le legano... rho = 0.5 e btmax  = 50 è un errore
% GRAVESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
% anche btmax = 30 nonè giusto perché vado lento
% con 50 backtracking e rho = 0.5 raggiungiamo eps... sbagliato
%
% AGGIUNGERE CONDIZIone di curvatura