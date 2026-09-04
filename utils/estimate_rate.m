function p = estimate_rate(xseq, gradf)
    % Estimation of the convergence rate based on the last three grandient norms.
    if isempty(xseq) || size(xseq,2) < 4
        p = NaN; return;
    end

    g3 = norm(gradf(xseq(:, end)));
    g2 = norm(gradf(xseq(:, end-1)));
    g1 = norm(gradf(xseq(:, end-2)));

    if any([g1,g2,g3] < 1e-14)
        p = NaN; return;
    end
    
    if abs(g3/g2) < 1e-8 || abs(g2/g1) < 1e-8
        p = NaN; return;
    end

    p = log(g3/g2) / log(g2/g1);
    if p < 0 || p > 3, p = NaN; end
end
