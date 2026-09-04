function H = hess_grad_fd_broyden31(x, k, type, rfun)
% HESS_GRAD_FD_BROYDEN31 Computes the FD Hessian using the FD gradient.
% It calculates the Hessian for Problem 31 applying a 5-coloring tecnique,
% using a 3-coloring tecnique on the gradient.
%
% INPUTS:
%   x    : current point
%   k    : power for the increment (e.g., 4, 8, 12)
%   type : 1 for constant step, 2 for relative step
%   rfun : handle to the exact residual function rvec(x)
%
% OUTPUT:
%   H    : fully approximated Hessian matrix

n = length(x);

% Handle function for the approximated gradient
gfun_fd = @(y) grad_fd_broyden31(y, k, type, rfun);

% Compute step sizes
hvec = steps_fd(x, k, type);

% 4. Preallocate sparse Hessian 
H = spalloc(n, n, 5*n);

% 5-coloring approach for the Hessian
for group = 1:5
    d = zeros(n, 1);
    indices = group:5:n; 

    % Apply perturbation to the current group
    d(indices) = hvec(indices); 

    % Evaluate FD gradient at the perturbed points (forward and backward)
    g_plus  = gfun_fd(x + d);
    g_minus = gfun_fd(x - d);

    % Compute the centered finite difference for the gradient
    delta_g = g_plus - g_minus;

    % Building the hessian matrix
    for j = indices
        row_start = max(1, j-2);
        row_end = min(n, j+2);

        H(row_start:row_end, j) = delta_g(row_start:row_end) / (2*hvec(j));
    end
end

% Taking the symmetric to correct numerical errors
H = (H + H') / 2;
end