clear; clc; close all;

%% ============================================================
%  TEST 1 — QUADRATICA CON MINIMO NOTO
%  f(x) = (1/2)*x'Ax - b'x,  A definita positiva
%  Minimo noto: x* = A\b
% ============================================================

disp("TEST 1: Quadratica con minimo noto (A PD)");

A = [4 1; 1 3];
b = [-1; 2];

f = @(x) 0.5*x'*A*x - b'*x;
gradf = @(x) A*x - b;
hessf = @(x) A;

x0 = [10; -5];

[xk,fk,grad_norm,k,~,~,tauk] = modified_newton_method2( ...
    x0,f,gradf,hessf,200,1e-8,1e-4,0.5,20,1e-3);
% function [xk,fk,gradfk_norm,k,xseq,btseq] = 
% modified_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

disp("Soluzione attesa:");
x_star = A\b

disp("Soluzione ottenuta:");
xk

disp("Errore norm(xk - x*) = ");
norm(xk - x_star)

disp("Numero iterazioni k");
k

disp(" Matrice correzioni per rendere Bk def.pos.");
sum(tauk,"all")

%% ============================================================
%  TEST 2 — ROSENBROCK 2D
%  Minimo noto: (1,1)
%  Hessiana non PD ovunque → test correzione τ
% ============================================================

disp("TEST 2: Rosenbrock (Hessiana non PD in molti punti)");

f = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
gradf = @(x) [ -400*x(1)*(x(2)-x(1)^2) - 2*(1-x(1));
               200*(x(2)-x(1)^2) ];
hessf = @(x) [ 1200*x(1)^2-400*x(2)+2  ,  -400*x(1);
               -400*x(1)               ,  200       ];

x0 = [-1.2; 1];

[xk,fk,grad_norm,k,~,~,tauk] = modified_newton_method( ...
    x0,f,gradf,hessf,2000,1e-7,1e-4,0.5,25,1e-3);

disp("Minimo vero: [1; 1]");
xk
norm(xk - [1;1])


disp("Numero iterazioni k");
k

disp(" Matrice correzioni per rendere Bk def.pos.");
sum(tauk,"all")

%% ============================================================
%  TEST 3 — HESSIANA INDEFINITA CHE DEVE ROMPERE LA CHOLESKY
%  f(x) = x1^2 - x2^2  → Hessiana = diag(2, -2) → NON PD
%  Verifica che τ venga aggiornato finché Bk è PD
% ============================================================

disp("TEST 3: Hessiana indefinita che rompe Cholesky (correzione τ obbligatoria)");

f = @(x) x(1)^2 - x(2)^2;
gradf = @(x) [2*x(1); -2*x(2)];
hessf = @(x) [2 0; 0 -2];  % Sempre indefinita

x0 = [1; 1];

[xk,fk,grad_norm,k,~,~,tauk] = modified_newton_method( ...
    x0,f,gradf,hessf,50,1e-8,1e-4,0.5,20,1e-3);

disp("Risultato ottenuto (non ci si aspetta convergenza a un minimo, solo stabilità):");
xk
fk
grad_norm

disp("Numero iterazioni k");
k

disp(" Matrice correzioni per rendere Bk def.pos.");
sum(tauk,"all")

%% ============================================================
%  TEST 4 — CASO PATOLOGICO: FUNZIONE NON DIFFERENZIABILE
%  f(x) = ||x|| → gradf mal definito in 0
%  Verifica robustezza (ci si aspetta rottura, ma non crash)
% ============================================================

disp("TEST 4: Funzione non differenziabile (patologica)");

f = @(x) norm(x);
gradf = @(x) x / norm(x);    % definizione che esplode in zero
hessf = @(x) eye(length(x)); % fittizia

x0 = [1e-12; -1e-12];   % vicino alla non differenziabilità

try
    [xk,fk,grad_norm,k] = modified_newton_method( ...
        x0,f,gradf,hessf,50,1e-6,1e-4,0.5,20,1);
    disp("Output ottenuto:");
    xk
catch ME
    disp("Errore catturato (corretto per funzione non regolare):");
    disp(ME.message);
end


%% ============================================================
%  TEST 5 — ARMIJO CHE NON SI SODDISFA MAI
%  f(x)=exp(100x1)+x2^2 → gradiente enorme → backtracking spinto
% ============================================================

disp("TEST 5: Armijo che fallisce spesso (backtracking estremo)");

f = @(x) exp(100*x(1)) + x(2)^2;
gradf = @(x) [100*exp(100*x(1)); 2*x(2)];
hessf = @(x) [10000*exp(100*x(1)), 0; 0, 2];

x0 = [-0.1; 10]; % grad enormi → Armijo dura molto

[xk,fk,grad_norm,k,~,btseq] = modified_newton_method( ...
    x0,f,gradf,hessf,80,1e-6,1e-4,0.2,20,1e-3);

disp("Numero max di backtracking raggiunto in qualche iterazione?");
max(btseq)


disp("Numero iterazioni k");
k

disp(" Matrice correzioni per rendere Bk def.pos.");
sum(tauk,"all")

%%
% Esempio: Rosenbrock
f = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
gradf = @(x) [ -400*x(1)*(x(2)-x(1)^2) - 2*(1-x(1));
                200*(x(2)-x(1)^2) ];
hessf = @(x) [ 1200*x(1)^2-400*x(2)+2 , -400*x(1);
                -400*x(1)             , 200 ];

x0 = [-1.2; 1];

[xk,fk,gn,k,xseq,btseq,tau_new,alphas,pks] = ...
    modified_newton_method2(x0,f,gradf,hessf,200,1e-8,1e-4,0.5,20,1e-3);

%Plot unico
figure(77); clf; hold on;
title('Modified Newton Method — Traiettoria, direzioni, step');
xlabel('x_1'); ylabel('x_2');

% Griglia contorno
[X,Y] = meshgrid(linspace(min(xseq(1,:))-1, max(xseq(1,:))+1, 300), ...
                 linspace(min(xseq(2,:))-1, max(xseq(2,:))+1, 300));

Z = arrayfun(@(i,j) f([i;j]), X, Y);
contour(X,Y,Z,40,'LineColor',[0.7 0.7 0.7]);

% Step-by-step plotting
for i = 1:k
    xk_prev = xseq(:,i);
    pk = pks(:,i);
    alpha = alphas(i);
    xk_new = xseq(:,i+1);

    plot(xk_prev(1), xk_prev(2), 'bo', 'MarkerFaceColor','b');
    quiver(xk_prev(1), xk_prev(2), pk(1), pk(2), 'r', 'LineWidth',1.3);
    plot([xk_prev(1) xk_new(1)], [xk_prev(2) xk_new(2)], 'm--', 'LineWidth',1.4);

    drawnow;
    pause(0.1); % opzionale (animazione)
end

legend({'Contorno f', ...
        'Iterati', ...
        'Direzioni p_k', ...
        'Step effettivi \alpha_k p_k'}, 'Location','best');
