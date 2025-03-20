% test_sfrmat5
% Peter Burns, 27 Feb 2023

% 1. GUI execution
% File, ROI, sampling and saving selction
sfrmat5;

% 2. Non-GUI execution
[~, total] = imageread;

io = 1;
del = 1;
npol = 5;
wflag = 0;
a = getroi(total);
% Default window, luminance weighting

[status, dat1, e, sfr50] = sfrmat5(io, del,a,5);

figure, plot(dat1(:,1),dat1(:,end))
xlabel('Frequency, cy/pixel')
ylabel('SFR')
axis([0 .7 0 1.05])

disp(['Sampling efficiency: ',num2str(e(1),3),'%'])
disp(['SFR50:      ',num2str(sfr50(1),3),' cy/pixel'])
