
%  This function executes the Modified Newton method across multiple
%  starting points to evaluate the robustness and efficiency of the 
%  provided oracles (Exact or FD).


% Output Data Structure
% res.fk:      Final objective function values for each start.
% res.gn:      Final gradient norms (used to check convergence).
% res.iters:   Number of iterations performed before stopping.
% res.time:    Wall-clock time spent on each optimization.
% res.success: Logical array (1 if ||g|| < tol, 0 otherwise).

function res = run_simulation(store, X0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta)

        num_pts = size(X0,2);
        
        % Preallocations.
        res = struct();
        res.success = false(num_pts,1);
        res.iters   = zeros(num_pts,1);
        res.gn      = zeros(num_pts,1);
        res.fk      = zeros(num_pts,1);
        res.time    = zeros(num_pts,1);
        
        for s = 1:num_pts

            % Selecting starting points.
            x0 = X0(:,s);
        
            tic;
            try
                [~,fk, gn, it] = modified_newton_method(x0, f, gradf, hessf, ...
                    kmax, tolgrad, c1, rho, btmax, beta);
            catch
                % In case of failure.
                fk = NaN; gn = Inf; it = 0;
            end
            t = toc;
            
            % Evaluate success based on the norm of the gradient.
            succ = (gn < tolgrad);
            
            % Storing.
            res.fk(s)      = fk;
            res.gn(s)      = gn;
            res.iters(s)   = it;
            res.time(s)    = t;
            res.success(s) = succ;
        
            startID = "xbar";
            if s > 1, startID = "rand" + (s-1); end
        
            fprintf("    %s: it=%d | ||g||=%.2e | f=%.3e | t=%.2fs | succ=%d\n", ...
                startID, it, gn, fk, t, succ);
        end
end