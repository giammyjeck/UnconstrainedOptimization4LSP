% dummy_grad e' necessario per plot senza richiamare gradf nel file di output
% Il vero gradf non e' serializzabile dentro display_results; in pratica la
% funzione display_results usa generate_gnorm_seq solo quando xseq e' presente
% e gia' popolato con punti coerenti: a quel punto serve la funzione grad
% reale. Per maggiore sicurezza, se l'utente vuole usare un gradf specifico,
% e' possibile modificare la chiamata passando il handle di gradf.
function g = dummy_grad(x)
    % Funzione placeholder: ritorna la norma come vettore per non andare in errore
    g = x; % non viene realmente usata se l'utente mantiene gradf disponibile
end