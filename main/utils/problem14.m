function [f, gradf, hessf, xbar] = problem14()
%PROBLEM14_DISCRETE_BVP  Problem 14 (Discrete boundary value problem).
% Returns function handles for objective, gradient, Hessian and the suggested start.
%
% F(x) = sum_{i=1}^n r_i(x)^2,  with x_0 = x_{n+1} = 0,  h = 1/(n+1)
% r_i(x) = 2x_i - x_{i-1} - x_{i+1} + (h^2/2) * (x_i + i h + 1)^3

f     = @F14;
gradf = @f14grad;
hessf = @f14Hess;

% Suggested starting point: x_i = i h (1 - i h)
xbar  = @(n) ((1:n)'/(n+1)) .* (1 - (1:n)'/(n+1));
end

% ---------------- Objective ----------------
function Fx = F14(x)
x = x(:); n = length(x); h = 1/(n+1);
z = x + h*(1:n)' + 1;
r = 2*x - [0; x(1:end-1)] - [x(2:end); 0] + (h^2/2) * (z.^3);
Fx = r' * r;   
end

% ---------------- Gradient ----------------
function g = f14grad(x)
x = x(:); n = length(x); h = 1/(n+1);
z = x + h*(1:n)' + 1;

f  = 2*x - [0; x(1:end-1)] - [x(2:end); 0] + (h^2/2) * (z.^3);

df = 2*ones(n,1) + (3*h^2/2) * (z.^2);          % dr_i / dx_i

% ∇F = 2 * J^T r, using the local dependence (neighbors only)
g = 2 * ( df .* f - [f(2:n); 0] - [0; f(1:n-1)] );
end

% ---------------- Hessian ----------------
function H = f14Hess(x)
x = x(:); n = length(x); h = 1/(n+1);
z = x + h*(1:n)' + 1;

f   = 2*x - [0; x(1:end-1)] - [x(2:end); 0] + (h^2/2) * (z.^3);
%first derivatives 
df  = 2*ones(n,1) + (3*h^2/2) * (z.^2);        
%second derivatives 
d2f = 3*h^2 * z;                               

% 5-diagonal structure:
%Main diagonal H(i,i) 
main = 2 * ( (df.^2 + 2*[0.5; ones(n-2,1); 0.5]) + f .* d2f );
% First sub/upper diagonal H(i,i+1)
off1 = -2 * (df(1:end-1) + df(2:end));
%secon sub/upper diagonal H(i,i+2) 
off2 =  2 * ones(n-2,1);
% H as a sparse matrix - 5n -6/ n^2 non zero elements
H = spdiags([[off2;0;0], [off1;0], main, [0;off1], [0;0;off2]], -2:2, n, n);
H = (H + H')/2;
end