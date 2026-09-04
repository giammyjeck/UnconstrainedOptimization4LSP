function g = trig_fd_case2_grad_only(x, k, mode)
% Estrae solo il gradiente da trig_fd_case2 (serve per passare
% un function handle @(x) trig_fd_case2_grad_only(x,k,mode) ai solver,
% che si aspettano gradf(x) con un solo output).
    [g, ~] = trig_fd_case2(x, k, mode);
end