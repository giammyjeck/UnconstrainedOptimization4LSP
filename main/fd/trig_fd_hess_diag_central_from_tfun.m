function H = trig_fd_hess_diag_central_from_tfun(tfun, x, k, mode)
% Hessiana di Trig16 è DIAGONALE (separabile).
% FD centrale per seconda derivata:
%   H_ii ≈ (t_i(x_i+h_i) - 2 t_i(x_i) + t_i(x_i-h_i)) / h_i^2

x = x(:);
n = length(x);
h = steps_fd(x,k,mode);

t0 = tfun(x);
tp = tfun(x + h);
tm = tfun(x - h);

d2 = (tp - 2*t0 + tm) ./ (h.^2);
H  = spdiags(d2, 0, n, n);
end