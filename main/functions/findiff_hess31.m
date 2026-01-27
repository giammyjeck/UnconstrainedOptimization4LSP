function hess = findiff_hess(f,x,h)
% This function computes noth the forward and the centerd formulas for
% approximating the jacobian matrix of a given F:Rn->Rm.

%Optimal value for h when computing the hessian: if i use a specific value
%for h when computing the gradient, then the h value for computing the
%hessian is the square root of the previous one: h_hess = square(h_grad).
%In this way i avoid the situation in which i use a very small value of h
%for the gradient that doesn't work for the hessian. 

    n = length(x);
    fx = f(x);
    hess = zeros(n,n);

    for i = 1:n
       xh_plus = x;
       xh_minus = x;
       xh_plus(i)= xh_plus+h;
       xh_minus(i)= xh_minus+h;
       hess(i,i)=(f(xh_plus)-2*fx+f(xh_minus))/(h^2);
       for j = 1:n
           hess(i,j)=(f(xh_plus)-2*fx+f(xh_minus))/(h^2);
       end
    end


end