function [grad] = findiff_grad(f,x,h,type)
% This function computes noth the forward and the centerd formulas for
% approximating the gradient of a given f.

    n = length(x);
    grad = zeros(n,1);
    fx = f(x);
    
    switch type
        case 'fw'
                for i=1:n
                    %costruisco prima i vettori in cui deve essere valutata
                    %la funzione e poi calcolo il gradiente. 
                    xh = x;
                    xh(i) = xh(i)+h;
                    %la funzione viene valutata nei vettori non nelle
                    %singole componenti
                    grad(i) = (f(xh)-fx)/h;  %forward formula
                end
        case 'c'
                for i=1:n
                    xh_plus = x;
                    xh_minus = x;
                    xh_plus(i) = xh_plus(i)+h;
                    xh_minus(i) = xh_minus(i)-h;
                    grad(i) = (f(xh_plus)-f(xh_minus))/(2*h);  %centered formula
                end
        otherwise
                error('The input must be either fw or c.')
    end
        
end

%Results on the test functions: if i use the machine precision as h then
%there's  not gonna be any difference between the exact gradient and the
%approximated one. 