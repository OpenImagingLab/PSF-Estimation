function [data] = ahamming(n, mid)
% [data] = ahamming(n, mid)
% function generates a general asymmetric Hamming-type window
% array. If mid = (n+1)/2 then the usual symmetric Hamming 
% window is returned
%  n = length of array
%  mid = midpoint (maximum) of window function
%  data = window array (nx1)
%
%  Author: Peter Burns, 1 Oct. 2008
%  Copyright (c) 2007 Peter D. Burns

data = zeros(n,1);
%
mid = mid+0.5;  % added 13 June 2019

wid1 = mid-1;
wid2 = n-mid;
wid = max(wid1, wid2);
pie = pi;
for i = 1:n
	arg = i-mid;
	data(i) = cos( pie*arg/(wid) );
end
data = 0.54 + 0.46*data;
