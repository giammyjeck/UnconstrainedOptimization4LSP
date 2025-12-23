clc
close all
clear

addpath(genpath('functions'));
addpath(genpath('test'));
load test_functions2.mat

%% Il codice è strutturato come segue:
% Prima definiamo alcuni parametri comuni alle due funzioni
% Poi un elenco di funzioni su cui è stato testato il corretto
% comportamento dei metodi, le funzioni dei test sono:
% 1- caso banale funzione quadratica
% 2- caso hessiana indefinita, non ho convergenza
% 3- caso f1 di test_functions2
% 4- caso f2 di ""
% 5- caso f3 di ""
%
% Alla fine un codice che permette di stampare il caso N=2 per dei plot



%% Parameter
% Common Parameter for MODIFIED

tolgrad = 1e-8;
c1 = 0.1;
rho = 0.5;
btmax = 20;
beta = 1e-3;
kmax = 100;

% common parameter for TRUNCATED

%% Convergenza su funzione quadratica
disp('Convergenza su funzione quadratica');
H = [2 0; 0 2];
f_quad = @(x) 0.5*x'*H*x;
grad_quad = @(x) H*x;
hess_quad = @(x) H;

% Metodo 1
[xk1,fk1,gradfk_norm1,k1,xseq1,btseq1,tau_new1,alphas1,pks1] = ...
    modified_newton_method([1;1], f_quad, grad_quad, hess_quad, kmax, tolgrad, c1, rho, btmax, beta);

% Metodo 2
[xk2,fk2,gradfk_norm2,k2,xseq2,btseq2,tau_new2,alphas2,pks2] = ...
    modified_newton_method2([1;1], f_quad, grad_quad, hess_quad, kmax, tolgrad, c1, rho, btmax, beta);

% Assert Metodo 1
assert(norm(xk1) < 1e-10, 'Test 1 fallito: minimo non raggiunto (method1)');
assert(k1 == 1, 'Test 1 fallito: Newton non converge in una iterazione (method1)');
assert(all(tau_new1(:) >= 0), 'Test 2 fallito: tau negativo trovato (method1)');
for i = 1:k1
    f_curr = f_quad(xseq1(:,i));
    f_next = f_quad(xseq1(:,i+1));
    grad_curr = grad_quad(xseq1(:,i));
    p_curr = pks1(:,i);
    alpha_curr = alphas1(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'Test 3 fallito: Armijo non soddisfatta (method1)');
end
assert(size(xseq1,2) == k1+1, 'Test 4 fallito: xseq dimensione errata (method1)');
assert(length(btseq1) == k1, 'Test 4 fallito: btseq dimensione errata (method1)');
assert(length(alphas1) == k1, 'Test 4 fallito: alphas dimensione errata (method1)');
assert(size(pks1,2) == k1, 'Test 4 fallito: pks dimensione errata (method1)');

% Assert Metodo 2
assert(norm(xk2) < 1e-10, 'Test 1 fallito: minimo non raggiunto (method2)');
assert(k2 == 1, 'Test 1 fallito: Newton non converge in una iterazione (method2)');
assert(all(tau_new2(:) >= 0), 'Test 2 fallito: tau negativo trovato (method2)');
for i = 1:k2
    f_curr = f_quad(xseq2(:,i));
    f_next = f_quad(xseq2(:,i+1));
    grad_curr = grad_quad(xseq2(:,i));
    p_curr = pks2(:,i);
    alpha_curr = alphas2(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'Test 3 fallito: Armijo non soddisfatta (method2)');
end
assert(size(xseq2,2) == k2+1, 'Test 4 fallito: xseq dimensione errata (method2)');
assert(length(btseq2) == k2, 'Test 4 fallito: btseq dimensione errata (method2)');
assert(length(alphas2) == k2, 'Test 4 fallito: alphas dimensione errata (method2)');
assert(size(pks2,2) == k2, 'Test 4 fallito: pks dimensione errata (method2)');

disp(['Metodo 1: xk = ', mat2str(xk1), ', fk = ', num2str(fk1)]);
disp(['Metodo 2: xk = ', mat2str(xk2), ', fk = ', num2str(fk2)]);
disp('Convergenza su funzione quadratica: DONE');

%% Caso funzione con Hessiano non positivo definito iniziale
disp('Funzione mock con Hessiano non positivo definito iniziale');
f_mock = @(x) x(1)^2 - x(2)^2 + x(1)*x(2);
grad_mock = @(x) [2*x(1)+x(2); -2*x(2)+x(1)];
hess_mock = @(x) [2 1; 1 -2];

[xk1,fk1,gradfk_norm1,k1,xseq1,btseq1,tau_new1,alphas1,pks1]= ...
    modified_newton_method([1;1], f_mock, grad_mock, hess_mock, kmax, tolgrad, c1, rho, btmax, beta);

[xk2,fk2,gradfk_norm2,k2,xseq2,btseq2,tau_new2,alphas2,pks2]= ...
    modified_newton_method2([1;1], f_mock, grad_mock, hess_mock, kmax, tolgrad, c1, rho, btmax, beta);

% Assert Metodo 1
assert(norm(gradfk_norm1) < tolgrad, 'Test fallito: gradiente finale troppo grande (method1)');
assert(all(tau_new1(:) >= 0), 'Test fallito: tau negativo trovato (method1)');
for i = 1:k1
    f_curr = f_mock(xseq1(:,i));
    f_next = f_mock(xseq1(:,i+1));
    grad_curr = grad_mock(xseq1(:,i));
    p_curr = pks1(:,i);
    alpha_curr = alphas1(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'Test fallito: Armijo non soddisfatta (method1)');
end
assert(size(xseq1,2) == k1+1, 'Test fallito: xseq dimensione errata (method1)');
assert(length(btseq1) == k1, 'Test fallito: btseq dimensione errata (method1)');
assert(length(alphas1) == k1, 'Test fallito: alphas dimensione errata (method1)');
assert(size(pks1,2) == k1, 'Test fallito: pks dimensione errata (method1)');

% Assert Metodo 2
assert(norm(gradfk_norm2) < tolgrad, 'Test fallito: gradiente finale troppo grande (method2)');
assert(all(tau_new2(:) >= 0), 'Test fallito: tau negativo trovato (method2)');
for i = 1:k2
    f_curr = f_mock(xseq2(:,i));
    f_next = f_mock(xseq2(:,i+1));
    grad_curr = grad_mock(xseq2(:,i));
    p_curr = pks2(:,i);
    alpha_curr = alphas2(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'Test fallito: Armijo non soddisfatta (method2)');
end
assert(size(xseq2,2) == k2+1, 'Test fallito: xseq dimensione errata (method2)');
assert(length(btseq2) == k2, 'Test fallito: btseq dimensione errata (method2)');
assert(length(alphas2) == k2, 'Test fallito: alphas dimensione errata (method2)');
assert(size(pks2,2) == k2, 'Test fallito: pks dimensione errata (method2)');

disp(['Metodo 1: xk = ', mat2str(xk1), ', fk = ', num2str(fk1)]);
disp(['Metodo 2: xk = ', mat2str(xk2), ', fk = ', num2str(fk2)]);
disp('Convergenza funzione con Hessiano non positivo definito iniziale: DONE');

%% Section f1
disp('Test funzione f1');
[xk1,fk1,gradfk_norm1,k1,xseq1,btseq1,tau_new1,alphas1,pks1] = ...
    modified_newton_method([1;1], f1, gradf1, Hessf1, kmax, tolgrad, c1, rho, btmax, beta);
[xk2,fk2,gradfk_norm2,k2,xseq2,btseq2,tau_new2,alphas2,pks2] = ...
    modified_newton_method2([1;1], f1, gradf1, Hessf1, kmax, tolgrad, c1, rho, btmax, beta);

% Assert Metodo 1
assert(norm(gradfk_norm1) < tolgrad, 'f1: gradiente finale troppo grande (method1)');
assert(all(tau_new1(:) >= 0), 'f1: tau negativo trovato (method1)');
for i = 1:k1
    f_curr = f1(xseq1(:,i));
    f_next = f1(xseq1(:,i+1));
    grad_curr = gradf1(xseq1(:,i));
    p_curr = pks1(:,i);
    alpha_curr = alphas1(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'f1: Armijo non soddisfatta (method1)');
end
assert(size(xseq1,2) == k1+1, 'f1: xseq dimensione errata (method1)');
assert(length(btseq1) == k1, 'f1: btseq dimensione errata (method1)');
assert(length(alphas1) == k1, 'f1: alphas dimensione errata (method1)');
assert(size(pks1,2) == k1, 'f1: pks dimensione errata (method1)');

% Assert Metodo 2
assert(norm(gradfk_norm2) < tolgrad, 'f1: gradiente finale troppo grande (method2)');
assert(all(tau_new2(:) >= 0), 'f1: tau negativo trovato (method2)');
for i = 1:k2
    f_curr = f1(xseq2(:,i));
    f_next = f1(xseq2(:,i+1));
    grad_curr = gradf1(xseq2(:,i));
    p_curr = pks2(:,i);
    alpha_curr = alphas2(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'f1: Armijo non soddisfatta (method2)');
end
assert(size(xseq2,2) == k2+1, 'f1: xseq dimensione errata (method2)');
assert(length(btseq2) == k2, 'f1: btseq dimensione errata (method2)');
assert(length(alphas2) == k2, 'f1: alphas dimensione errata (method2)');
assert(size(pks2,2) == k2, 'f1: pks dimensione errata (method2)');

disp(['Metodo 1: xk = ', mat2str(xk1), ', fk = ', num2str(fk1)]);
disp(['Metodo 2: xk = ', mat2str(xk2), ', fk = ', num2str(fk2)]);
disp('Convergenza su f1: DONE');

%% Section f2
disp('Test funzione f2');
[xk1,fk1,gradfk_norm1,k1,xseq1,btseq1,tau_new1,alphas1,pks1] = ...
    modified_newton_method([1;1], f2, gradf2, Hessf2, kmax, tolgrad, c1, rho, btmax, beta);
[xk2,fk2,gradfk_norm2,k2,xseq2,btseq2,tau_new2,alphas2,pks2] = ...
    modified_newton_method2([1;1], f2, gradf2, Hessf2, kmax, tolgrad, c1, rho, btmax, beta);

% Assert Metodo 1
assert(norm(gradfk_norm1) < tolgrad, 'f2: gradiente finale troppo grande (method1)');
assert(all(tau_new1(:) >= 0), 'f2: tau negativo trovato (method1)');
for i = 1:k1
    f_curr = f2(xseq1(:,i));
    f_next = f2(xseq1(:,i+1));
    grad_curr = gradf2(xseq1(:,i));
    p_curr = pks1(:,i);
    alpha_curr = alphas1(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'f2: Armijo non soddisfatta (method1)');
end
assert(size(xseq1,2) == k1+1, 'f2: xseq dimensione errata (method1)');
assert(length(btseq1) == k1, 'f2: btseq dimensione errata (method1)');
assert(length(alphas1) == k1, 'f2: alphas dimensione errata (method1)');
assert(size(pks1,2) == k1, 'f2: pks dimensione errata (method1)');

% Assert Metodo 2
assert(norm(gradfk_norm2) < tolgrad, 'f2: gradiente finale troppo grande (method2)');
assert(all(tau_new2(:) >= 0), 'f2: tau negativo trovato (method2)');
for i = 1:k2
    f_curr = f2(xseq2(:,i));
    f_next = f2(xseq2(:,i+1));
    grad_curr = gradf2(xseq2(:,i));
    p_curr = pks2(:,i);
    alpha_curr = alphas2(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'f2: Armijo non soddisfatta (method2)');
end
assert(size(xseq2,2) == k2+1, 'f2: xseq dimensione errata (method2)');
assert(length(btseq2) == k2, 'f2: btseq dimensione errata (method2)');
assert(length(alphas2) == k2, 'f2: alphas dimensione errata (method2)');
assert(size(pks2,2) == k2, 'f2: pks dimensione errata (method2)');

disp(['Metodo 1: xk = ', mat2str(xk1), ', fk = ', num2str(fk1)]);
disp(['Metodo 2: xk = ', mat2str(xk2), ', fk = ', num2str(fk2)]);
disp('Convergenza su f2: DONE');

%% Section f3
disp('Test funzione f3');
[xk1,fk1,gradfk_norm1,k1,xseq1,btseq1,tau_new1,alphas1,pks1] = ...
    modified_newton_method([1;1], f3, gradf3, Hessf3, kmax, tolgrad, c1, rho, btmax, beta);
[xk2,fk2,gradfk_norm2,k2,xseq2,btseq2,tau_new2,alphas2,pks2] = ...
    modified_newton_method2([1;1], f3, gradf3, Hessf3, kmax, tolgrad, c1, rho, btmax, beta);

% Assert Metodo 1
assert(norm(gradfk_norm1) < tolgrad, 'f3: gradiente finale troppo grande (method1)');
assert(all(tau_new1(:) >= 0), 'f3: tau negativo trovato (method1)');
for i = 1:k1
    f_curr = f3(xseq1(:,i));
    f_next = f3(xseq1(:,i+1));
    grad_curr = gradf3(xseq1(:,i));
    p_curr = pks1(:,i);
    alpha_curr = alphas1(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'f3: Armijo non soddisfatta (method1)');
end
assert(size(xseq1,2) == k1+1, 'f3: xseq dimensione errata (method1)');
assert(length(btseq1) == k1, 'f3: btseq dimensione errata (method1)');
assert(length(alphas1) == k1, 'f3: alphas dimensione errata (method1)');
assert(size(pks1,2) == k1, 'f3: pks dimensione errata (method1)');

% Assert Metodo 2
assert(norm(gradfk_norm2) < tolgrad, 'f3: gradiente finale troppo grande (method2)');
assert(all(tau_new2(:) >= 0), 'f3: tau negativo trovato (method2)');
for i = 1:k2
    f_curr = f3(xseq2(:,i));
    f_next = f3(xseq2(:,i+1));
    grad_curr = gradf3(xseq2(:,i));
    p_curr = pks2(:,i);
    alpha_curr = alphas2(i);
    armijo = f_curr + c1*alpha_curr*(grad_curr'*p_curr);
    assert(f_next <= armijo + 1e-12, 'f3: Armijo non soddisfatta (method2)');
end
assert(size(xseq2,2) == k2+1, 'f3: xseq dimensione errata (method2)');
assert(length(btseq2) == k2, 'f3: btseq dimensione errata (method2)');
assert(length(alphas2) == k2, 'f3: alphas dimensione errata (method2)');
assert(size(pks2,2) == k2, 'f3: pks dimensione errata (method2)');

disp(['Metodo 1: xk = ', mat2str(xk1), ', fk = ', num2str(fk1)]);
disp(['Metodo 2: xk = ', mat2str(xk2), ', fk = ', num2str(fk2)]);
disp('Convergenza su f3: DONE');

%% CODICE PER IL PLOT CASO N=2

%Plot unico
figure(77); clf; hold on;
title('Modified Newton Method — Traiettoria, direzioni, step');
xlabel('x_1'); ylabel('x_2');

% Griglia contorno
[X,Y] = meshgrid(linspace(min(xseq(1,:))-1, max(xseq(1,:))+1, 300), ...
                 linspace(min(xseq(2,:))-1, max(xseq(2,:))+1, 300));

Z = arrayfun(@(i,j) f_3([i;j]), X, Y);
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



