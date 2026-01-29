function H = hess_fd_from_grad_banded(gradf, x, k, mode, bw)
% Hessiana via FD (forward) del gradiente:
%   H(:,j) ≈ (g(x + h_j e_j) - g(x)) / h_j
%
% ma costruita SOLO nella banda |i-j| <= bw
% e usando "coloring" 1D per ridurre le valutazioni:
%   m = 2*bw+1 valutazioni di gradiente (oltre g(x))

x = x(:);
n = length(x);

hvec = steps_fd(x,k,mode);
g0   = gradf(x);

m = 2*bw + 1;
nnz_est = n*(2*bw+1);

I = zeros(nnz_est,1);
J = zeros(nnz_est,1);
V = zeros(nnz_est,1);
ptr = 0;

for r = 1:m
    idx = r:m:n;

    e = zeros(n,1);
    e(idx) = hvec(idx);

    gr = gradf(x + e);
    dg = gr - g0;

    for t = 1:numel(idx)
        j = idx(t);
        denom = hvec(j);

        ii = (max(1,j-bw):min(n,j+bw))';

        ptr2 = ptr + numel(ii);
        I(ptr+1:ptr2) = ii;
        J(ptr+1:ptr2) = j;
        V(ptr+1:ptr2) = dg(ii) / denom;
        ptr = ptr2;
    end
end

I = I(1:ptr); J = J(1:ptr); V = V(1:ptr);
H = sparse(I,J,V,n,n);

% simmetrizza (Hess vera è simmetrica; qui riduciamo rumore numerico)
H = (H + H')/2;
end
