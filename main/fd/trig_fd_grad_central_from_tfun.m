%% ========================================================================
%  CASE 2 - TRIG16: grad e Hess diagonale con FD centrali su tfun
% ========================================================================
function g = trig_fd_grad_central_from_tfun(tfun, x, k, mode)
% Trig16 è separabile:
%   F(x) = sum_i t_i(x_i)
% quindi:
%   g_i = d/dx_i t_i(x_i)
%
% FD centrale:
%   g_i ≈ (t_i(x_i+h_i) - t_i(x_i-h_i)) / (2 h_i)

x = x(:);
h = steps_fd(x,k,mode);

tp = tfun(x + h);   % vettore (t_i(x_i+h_i))
tm = tfun(x - h);   % vettore (t_i(x_i-h_i))

g  = (tp - tm) ./ (2*h);
end

