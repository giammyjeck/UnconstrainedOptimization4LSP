function [f_h, g_h, h_h] = get_broyden_functions(p)
% GET_BROYDEN_FUNCTIONS Restituisce gli handle per f, grad e hess
% Uso: [f, g, h] = get_broyden_functions(7/3);

    % Restituiamo i tre handle che puntano alle funzioni scritte sotto
    f_h = @(x) broyden_f(x);
    g_h = @(x) broyden_grad(x);
    h_h = @(x) broyden_hess(x);

    % --- EVALUATION OF f ---
    function f_val = broyden_f(x)
        phi = calculate_phi(x); 
        f_val = sum(abs(phi).^p);
    end

    % --- GRADIENT ---
    function g = broyden_grad(x)
        phi = calculate_phi(x);
        d_outer = p * (abs(phi).^(p-1)) .* sign(phi);
        
        g = d_outer .* (3 - 4*x);
        g(1:end-1) = g(1:end-1) - d_outer(2:end);
        g(2:end)   = g(2:end)   - 2*d_outer(1:end-1);
    end

    % --- SPARSE HESSIAN MATRIX ---

    % Per gestire n = 1e5, è obbligatorio usare matrici SPARSE (comando 'spdiags') 
    % per l'Hessiana, altrimenti la memoria RAM si esaurirà immediatamente.
    % La struttura tridiagonale della funzione di Broyden permette al metodo di 
    % Newton di risolvere il sistema Bk*pk = -gradfk in tempo lineare O(n).
    % I parametri scelti (c1=1e-4, rho=0.5) sono standard per garantire la 
    % convergenza globale tramite backtracking di Armijo. 
    % Il parametro beta=1e-3 assicura che Bk sia ben condizionata, 
    % evitando errori numerici nel solutore diretto R\(R'\(-gradfk)).
    function H = broyden_hess(x)
        n = length(x);
        phi = calculate_phi(x);
        
        d2_outer = p * (p-1) * (abs(phi).^(p-2));
        d1_outer = p * (abs(phi).^(p-1)) .* sign(phi);
        
        % Main diagonal
        d = d2_outer .* (3 - 4*x).^2 - 4 * d1_outer;
        d(1:end-1) = d(1:end-1) + d2_outer(2:end);
        d(2:end)   = d(2:end)   + 4 * d2_outer(1:end-1);
        
        % Secondary diagonals (just computing the upper ones because the
        % hessian is symmetric
        u = (d2_outer(1:end-1) .* (3 - 4*x(1:end-1)) * (-2)) + ...
            (d2_outer(2:end) .* (-1) .* (3 - 4*x(2:end)));
        
        % Computing the tridiagonal hessian (sparse)
        H = spdiags([[u; 0], d, [0; u]], [-1,0,1], n, n);
        H = (H + H')/2;
    end
    
    % La funzione di Broyden è composta da tanti termini phi_i elevati a p,
    % calcola i singoli termini della somma
    function phi = calculate_phi(x)
        x_prev = [0; x(1:end-1)];
        x_next = [x(2:end); 0];
        phi = (3 - 2*x).*x - x_prev - 2*x_next + 1;
    end
end