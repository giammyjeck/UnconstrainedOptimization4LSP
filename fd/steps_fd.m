function hvec = steps_fd(x, k, type)
% This function computes the finite difference step size vector.
%
% INPUTS:
%   x    : current point
%   k    : power for the increment (e.g., 4, 8, 12)
%   type : 1 -> 'h'  for constant step h_i = 10^-k
%          2 -> 'hi' for relative step h_i = 10^-k * |x_i|
%
% OUTPUT:
%   hvec : vector of step sizes

    x = x(:);
    h = 10^(-k);
    
    if type == 1
        hvec = h * ones(length(x), 1);
    else
        hvec = h * abs(x);
        hvec(hvec == 0) = h; % safeguard
	hvec(hvec < 1e-14) = 1e-14;
    end
    
end
