%% ========================================================================
% TASK 3 - CASE 1 (solo n=2, k=4) per plot 2D
% ========================================================================
clear; clc; close all;

addpath(genpath("C:\Users\Utente\Desktop\Corsi\Numerical optimization for large scale problems and Stochastic Optimization\NumericalO4LSP\main"));

%% --- seed ---
seed = 346710; 
rng(seed,"twister");

%% --- problema ---
[f, grad_exact, hess_exact, xbarfun] = problem_broyden31();
pname = "problem_broyden31";

%% --- parametri Modified Newton ---
kmax    = 20;
tolgrad = 1e-6;
c1      = 1e-4;
rho     = 0.3;
btmax   = 5;
max_cg  = 5;

%% --- starting points (1 standard + 5 random) ---
n = 2;  
xbar = xbarfun(n);
X0   = [xbar, xbar + (2*rand(n,5)-1)];  % 6 start points

%% --- FD parameters ---
kfd  = 8;
mode = 'h';
bw   = 2; % Trig16: diagonale
Hfd = @(x) hess_fd_from_grad_banded(grad_exact, x, kfd, mode, bw);

%% --- RUN Modified Newton ---
store = true; % non salviamo xseq nel run
res_fd = run_6starts_success(store, X0, f, grad_exact, Hfd, ...
            kmax, tolgrad, c1, rho, btmax, max_cg);

%% --- prepara struttura per la funzione outputs ---
labels = {};
res_method = struct();



for s = 1:size(X0,2)
    labels{end+1} = sprintf('n2_pt%d', s);
    % salva la traiettoria dello start s
    res_method.(labels{end}).xseq = res_fd(s).xseq;  
end

%% --- cartella per grafici ---
figdir = "out_task3_case1_success";
if ~exist(figdir,"dir"), mkdir(figdir); end

%% --- genera plot 2D ---
%method = 'ModifiedNewton';
method = 'TruncatedNewton';

outputs(n, labels, res_method, f, method, figdir, pname);


