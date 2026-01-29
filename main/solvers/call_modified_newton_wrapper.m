%% ========================================================================
%  Wrapper robusto: modified_newton_method con o senza argomento "store"
% ========================================================================
function [fk, gn, it] = call_modified_newton_wrapper(x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta, store)
% La tua modified_newton_method potrebbe avere firma:
%   modified_newton_method(x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta)
% oppure includere un parametro "store".
%
% Quindi:
% 1) provo con store
% 2) se MATLAB dice "troppi input", riprovo senza

try
    % prova con store
    [~, fk, gn, it] = modified_newton_method(x0, f, gradf, hessf, ...
        kmax, tolgrad, c1, rho, btmax, beta, store);
    return
catch ME
    if contains(ME.message, "Too many input") || contains(ME.message, "nargin")
        % riprova senza store
        [~, fk, gn, it] = modified_newton_method(x0, f, gradf, hessf, ...
            kmax, tolgrad, c1, rho, btmax, beta);
        return
    else
        rethrow(ME)
    end
end
end
