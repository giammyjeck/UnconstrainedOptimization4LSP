function hvec = steps_fd(x, k, mode)
% mode:
%   'h'  : h_i = 10^-k
%   'hi' : h_i = 10^-k * max(|x_i|,1)
% (max(|x_i|,1) è il safeguard "semplice" che evita passi quasi zero)

    x = x(:);
    h = 10^(-k);
    
    if strcmp(mode,'hi')
        hvec = h * max(abs(x), 1);
    else
        hvec = h * ones(size(x));
    end
    
end
