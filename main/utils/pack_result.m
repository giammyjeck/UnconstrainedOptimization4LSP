%% --- Funzioni di utilita' ---
function r = pack_result(time, iters, gnorm, xseq, rate, flag)
    r.time = time;
    r.iters = iters;
    r.gnorm = gnorm;
    r.xseq = xseq;
    r.rate = rate;
    r.flag = flag;
end
