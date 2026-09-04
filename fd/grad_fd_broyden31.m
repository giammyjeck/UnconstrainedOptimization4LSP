function g = grad_fd_broyden31(x, k, type, rfun)
% GRAD_FD_BROYDEN31 Computes the FD gradient for Problem 31.
% It exploits the tridiagonal structure of the Jacobian, computing
% it with centered finite differences (3-coloring approach).
%
% INPUTS:
%   x    : current point
%   k    : power for the increment (e.g., 4, 8, 12)
%   type : 1 for constant step, 2 for relative step
%   rfun : handle to the residual function rvec(x)
%
% OUTPUT:
%   g    : approximated gradient

n = length(x);

% Compute the step size 
hvec = steps_fd(x, k, type);

% Preallocate sparse Jacobian (tridiagonal -> max 3*n non-zeros)
J = spalloc(n, n, 3*n);

% 3-coloring approach -> variables spaced by 3 do not overlap
for group = 1:3
    d = zeros(n, 1);
    indices = group:3:n;
    d(indices) = hvec(indices); % Apply perturbation only to the current group

    % Evaluate residuals at the perturbed x
    r_plus = rfun(x + d);
    r_minus = rfun(x-d);	

    % Compute the finite difference for the residuals
    delta_r = r_plus - r_minus;

    % Building the Jacobian 
    for j = indices
        row_start = max(1, j-1);
        row_end = min(n, j+1);

        J(row_start:row_end, j) = delta_r(row_start:row_end) / (2*hvec(j));
    end
end

% Compute the final approximated gradient: g = J' * r
r_x=rfun(x);
g = J' * r_x;
end