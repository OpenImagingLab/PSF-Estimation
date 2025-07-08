function [w] = tukey2(n, alpha, mid)
% Asymmetrical Tukey Tapered Cosine Window
% w = tukey2(n,alpha, mid) returns an n-point Tukey window with
% center at mid. Tukey window parameter is alpha
%
% n     = length of window vector
% alpha = window shape parameter, default = 1 (Hann window)
% mid   = center of window [1:nl], default = n/2 (symmetrical window)
%
% Needs: tukey function (included in this file)
%
% Author: Peter Burns, 30 Oct. 2019, updated 16 March 2022

if nargin<3
    mid = n/2;
end
if nargin <2
    alpha = 1;
end
if n<3;
    w = ones(n,1);
    return
end

m1 = n/2;
m2 = mid;
m3 = n-mid;
mm = max(m2,m3);
n2 = 2*mm;
n2 = round(n2);

w = tukey(n2, alpha);

if mid>=m1
    w = w(1:n);
else
    w = w(1+end-n:end);
end

function w = tukey(n, alpha)
% Tukey Window
%    tukey(n, alpha) returns a symmetric N point Tukey window with
%    parameter alpha.  The default value of alpha is 0.5.
%
if nargin < 2
   alpha = 0.5;
end
if (n == 1)
   w = 1;
   return
end
if (alpha == 0)
    w = ones(n,1);
    return
end
M = (n-1)/2;
w = zeros(n,1);
for k = 0:M
    if (k <= alpha*M)
        w(k+1) = 0.5*(1 + cos(pi*(k/alpha/M - 1)));
    else
        w(k+1) = 1;
    end
    w(n-k) = w(k+1);
end
w = w(:);

