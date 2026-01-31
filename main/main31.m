%% Main Script – Contour + Trajectories (n = 2)
clear; clc; close all;

seed = 346710;
rng(seed);
addpath(genpath(pwd))

% Problem definition
[f, gradf, hessf, xbar_gen] = problem_broyden31();

% Parameters
dims     = 2;          % SOLO 2D
kmax     = 20;
tolgrad  = 1e-6;
c1       = 1e-4;
rho      = 0.3;
btmax    = 5;
max_cg   = 5;

method = 'TruncatedNewton';
%method = 'ModifiedNewton';

figdir = 'figures';
pname  = 'Problem31';

if ~exist(figdir,'dir')
    mkdir(figdir);
end

%% Starting points
x0_standard = xbar_gen(2);
x0_random   = (x0_standard - 1) + 2 * rand(2,5);
all_x0      = [x0_standard, x0_random];

%% Structures expected by outputs()
labels = {};
res_method = struct();

for s = 1:size(all_x0,2)
    x0 = all_x0(:,s);
    label = sprintf('n2_pt%d', s);
    labels{end+1} = label; %#ok<SAGROW>

    [~, ~, gnorm, k, xseq, ~, ~, ~] = truncated_newton_method( ...
        x0, f, gradf, hessf, ...
        kmax, tolgrad, c1, rho, btmax, max_cg);

    % Store ONLY what is needed by outputs
    res_method.(label).xseq  = xseq;
    res_method.(label).k     = k;
    res_method.(label).gnorm = gnorm;
end

%% Call the plot (IDENTICAL to what you want)
outputs(dims, labels, res_method, f, method, figdir, pname);
%%
