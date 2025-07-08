function [status, dat, e, sfr50, fitme, esf_all, nbin, del2] = sfrmat5(io, del, a, npol, wflag, weight)
% MatLab function: sfrmat5 (v1) ISO 12233 4th editincolon: Slanted-edge analysis
%                               with polynomial edge fitting, 
% [status, dat, e, fitme, esf, nbin, del2] = sfrmat5(io, del, a, npol, wflag, weight);
%       From a selected edge area of an image, the program computes
%       the ISO slanted edge SFR. Input file can be single or
%       three-record file. Many image formats are supported. The image
%       is displayed and a region of interest (ROI) can be chosen, or
%       the entire field will be selected by not moving the mouse
%       when defining an ROI. Either a vertical or horizontal edge
%       feature. 
%  Input arguments:
%      io  (optional)
%        0 = (default) GUI(file,ROI sampling selection)
%        1 = Non GUI usage with supplied image data array
%      del (optional) sampling interval in mm or pixels/inch
%          If dx < 1 it is assumed to be sampling pitch in mm
%          If io = 1 (see below, no GUI) and del is not specified,
%          it is set equal to 1, so frequency is given in cy/pixel.
%      a   (required if io =1) an (n x m) or (n x m x 3) array of data
%      npol = (optional) order of polynomial fit to edge [1-5]
%           = 1 linear fit as per previous ISO 12233 Standard
%           = 2 second-order fit
%           = 5 (default) fifth-order fit
%      wflag = (optional) smoothing window
%            =  0 (default) Tukey smoothing window
%            = 1 Hamming window
%     weight = (optional) r,g,b weights used to compute luminance record
%              default = [0.213   0.715   0.072]
%      
% Returns: 
%       status = 0 if normal execution
%       dat  = computed sfr data
%       e    = sampling efficiency
%       SFR50 = Frequency where SFR = 50 %
%       fitme = coefficients for the polynomial equations for the fit to
%               edge locations for each color-record. For a 3-record
%               data file, fitme is a (npol+1 x 3) array, with the last column
%               being the color misregistration value (with green as 
%               reference).
%       esf  =  supersampled edge-profile array
%       nbin = binning factor used
%       del2 = sampling interval for esf, from which the SFR spatial
%              frequency sampling is was computed. This will be 
%              approximately 4 times the original image sampling.
% NOTE: The edge feature used for e-SFR analysis should cross either both the 
%       top and bottom, or both the left and right margins of the Region
%       of Intertest (ROI).
%EXAMPLE USAGE:
% sfrmat5     file and ROI selection and 
% sfrmat5(0, del) = GUI usage with del as default sampling in mm 
%                   or dpi 
% sfrmat5(0, del, [], 3) = GUI usage with del as default sampling
%                   in mm or dpi and third-order edge fitting
% sfrmat5(1, [],dat) = non-GUI usage for data array, dat, with default
%                   sampling and weights aplied (del =1)
% [status, dat, fitme] = sfrmat5(1, del, a, [], [],);
%                   sfr and edge locations, are returned for data
%                   array a, with specified sampling interval and luminance
%                   weights
%
% NOTE: The edge feature in the test image must pass through two corresponding
% edges of the region of interest (ROI) used for the analysis. The edge must
% pass through either the top and bottom, or the left and right margins of
% the ROI.
% legg
%Author: Peter Burns, Based on sfrmat4, adapted for polynomial edge fit
%         27 Oct. 2020 Updated with support for *.ptw and *.ptm image files
%                      from FLIR (forward-looking infrared) cameras.
%         29 June 2021 Tested for near-zero and near-45 degree edges, and 
%                      prepared for inclusion in ISO 12233 edition 4
%         28 Feb. 2023 posting following the publishing of ISO 12233 ed.4
% Copyright (c) 2009-2023 Peter D. Burns, pdburns@ieee.org
%******************************************************************
pflag = 0; % Used for diagnostic plotting
 
home = pwd;                  
name =    'sfrmat5';
version = '1';
when =    '28 Feb. 2023';

% Default settings             
guidefweight = [0.213   0.715   0.072]';
defnpol = 5;
oename = 'none';
nbin = 4;
% Default Tukey window:  wflag = 0
defalpha = 1;% Default Tukey window parameter, alpha = 1
maxalpha = 1;
minalpha = .5;

switch nargin

    case 0
     io =0;
     del =1;
     npol = defnpol;
     wflag = 0;
     weight = guidefweight;
     alpha = defalpha;
    case 1
      if isempty(io) ==1
          io =0;
      end
      del = 1;
      npol = defnpol;
      wflag = 0;
      weight = guidefweight;
      alpha = defalpha; 
    case 2
     if isempty(io) == 1
         io = 0;
     end
     if isempty(del) == 1
         del = 1;
     end
     npol = defnpol;
     wflag = 0;
     weight = guidefweight;
     alpha = defalpha; 
    case 3
      if isempty(io) == 1
         io = 0;
      end
      if isempty(del) == 1
         del = 1;
      end 
       npol = defnpol;   
       wflag = 0;
       weight = guidefweight;
       alpha = defalpha; 
    case 4
      if isempty(io) == 1
         io = 0;
      end
      if isempty(del) == 1
         del = 1;
      end
      if isempty(npol) == 1
         npol = defnpol;
      end
      a = double(a);
      wflag = 0;
      weight = guidefweight;
      alpha = defalpha; 
    case 5
      if isempty(io) == 1
         io = 0;
      end
       if isempty(del) == 1
         del = 1;
      end
      if isempty(npol) == 1
         npol = defnpol;
      end
      a = double(a);
      weight = guidefweight;
      alpha = defalpha; 
     case 6
      if isempty(io) == 1
         io = 0;
      end
       if isempty(del) == 1
         del = 1;
      end
      if isempty(npol) == 1
         npol = defnpol;
      end
       a = double(a);
      
      if isempty(weight) == 1
           weight = guidefweight;
      else
          wsize = size(weight);
          if isequal(wsize,[1, 3])
             weight = guidefweight;
          end
      end
      alpha = defalpha;
%     otherwise
%      disp('Incorrect number or arguments. There should be 1 - 6');
%      status = 1;
%      return

end
if isempty(npol) == 1
 npol = defnpol;
end
if isempty(weight)
    weight = guidefweight;
end
if isempty(alpha)
    alpha = defalpha;
end

% if exist('npol','var') ~=1
%     norder = input('Edge fit order? [5]');
%     if isempty(norder)==1
%         npol = 5;
%     else
%         npol = norder;
%     end
% end
if npol>5
    disp('* Edge fit order must be 5 or less *')
    npol = 5;
end
% Clip (restrict) alpha to the range [minalpha, maxalpha]
 alpha = clip(alpha, minalpha, maxalpha);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if io ~= 1    
    swin = splash(name, version, when);    
    drawnow;
    [status, atemp, ~, pathname, f] = imageread;
    close(swin); drawnow;
    if status~=0
        disp('No file selected. To try again type: > sfrmat5');
        status = 1;
        return;
    end
    filename = [pathname,f];

    [~, ~, ncol] = size(atemp);

    % input sampling and luminance weights
    if ncol==1
        [del, npol] = inbox2(del,npol);
    else 
        [del, weight, npol] = inbox4(del, guidefweight, npol); 
    end

    % used for plotting and listing
    if del==1
        sunit = 'pixel';
        funit =  'cy/pixel';
    else
        sunit = 'mm';
        funit = 'cy/mm';
    end
    sinfo.sunit = sunit;
    sinfo.funit = funit;
    
    cname = class(atemp);
    if strcmp(cname(1:5),'uint1')   % uint16
        smax = 2^16-1;
    elseif strcmp(cname(1:5),'uint8')
        smax = 255;
    else
        smax = 1e10;
    end
% 1. extract Region of interest
    [a, roi] = getroi(atemp);
    a = double(a);
    clear atemp
    [nlow, nhigh, cstatus] = clipping(a, 0, smax, 0.005);
    if cstatus ~=1 
     disp('Fraction low data');
     disp(nlow);
     disp('Fraction high data');
     disp(nhigh);
    end

else                     % when io = 1
    a= double(a);
    samp = zeros(1,3);
    samp(1) = nbin;
    samp(2) = del; % Image sampling
    if del > 1
       del = 25.4/del;  % Assume input was in DPI convert to pitch in mm
    end

end
[nlin, npix, ncol] = size(a);

% Form luminance record using the weight vector for red, green and blue
if ncol ==3
    lum = zeros(nlin, npix);
    lum = weight(1)*a(:,:,1) + weight(2)*a(:,:,2) + weight(3)*a(:,:,3); 
    cc = zeros(nlin, npix*4);
    cc = [ a(:, :, 1), a(:, :, 2), a(:,:, 3), lum];
    cc = reshape(cc,nlin,npix,4);

    a = cc;
    clear cc;
    clear lum;
    ncol = 4; 
end

% Rotate horizontal edge so it is vertical
rflag = 0;
 [a, nlin, npix, rflag] = rotatev2(a);  %based on data values
loc = zeros(ncol, nlin);

lowhi = 0;
fil1 = [0.5 -0.5];
fil2 = [0.5 0 -0.5];
% We Need 'positive' edge
tleft  = sum(sum(a(:,      1:5,  1),2));
tright = sum(sum(a(:, npix-5:npix,1),2));
if tleft>tright
    lowhi = 1;
    fil1 = [-0.5 0.5];
    fil2 = [-0.5 0 0.5];
end
% Test for low contrast edge;
 test = abs( (tleft-tright)/(tleft+tright) );
 if test < 0.2
    disp(' ** WARNING: Edge contrast is less that 20%, this can');
    disp('             lead to high error in the SFR measurement.');
 end

fitme = zeros(ncol, npol+1);
fitme1 = zeros(ncol, 2);
slout = zeros(ncol, 1);

% Smoothing window for first part of edge location estimation - 
%  to be used on each line of ROI  (% Symmetric window)
if wflag~=0
    win1 = ahamming(npix, (npix+1)/2);
    disp('Hamming window used')
else
    win1 = tukey2(npix, alpha, (npix+1)/2);
    % This step makes the edge finding more stable
    win1 = 0.95*win1 + 0.05;
end

for color=1:ncol                      % Loop for each color
    
    if pflag ==1
        pname = ' ';
        if ncol~=1
            pname =[' Red '
           'Green'
           'Blue '
           ' Lum '];
        end
        figure(1);
        ht = 400;
        rat = npix/nlin;
        if rat<=0.3
            rat = 0.3;
        end
        wid = round(ht*rat);
        pos = [25 0 0 0];
        pos(3) = wid;
        pos(4) = ht;
        set(gcf,'Position', pos);
        imagesc1(a(:,:, color));
        colormap('gray');
        title(pname(color,:));
    end % if pflag == 1

    if pflag ==1
        figure;
        mesh( a(:,:,color) ), colormap('default'), title(pname(color,:));
        xlabel('pixel'), ylabel('line'), zlabel('value'); % Pause to inspect data      
        
    end  % if pflag == 1
    
% 3. 1-D derivative  
    c = deriv1(a(:,:,color), nlin, npix, fil1);

% 4. Apply window and compute central locations of edge for each row
%    Compute centroid for derivative array for each line in ROI. 
%    NOTE WINDOW array 'win1'
    for n=1:nlin
       loc(color, n) = centroid( c(n, 1:npix )'.*win1) - 0.5;   % -0.5 shift for FIR phase
    end    
    fitme(color,:) = findedge2(loc(color,:), nlin, npol); 

    place = zeros(nlin,1);
    if wflag ~=0
        for n=1:nlin 
            place(n) = polyval(fitme(color,:), n-1);  
            win2 = ahamming(npix, place(n));
            loc(color, n) = centroid( c(n, 1:npix )'.*win2) -0.5;
        end
    else  
      for n=1:nlin 
            place(n) = polyval(fitme(color,:), n-1); 
             win2 = tukey2(npix, alpha, place(n));
             % This step makes the edge finding more stable
             win2 = 0.95*win2 + 0.05;
            loc(color, n) = centroid( c(n, 1:npix )'.*win2) -0.5;
      end   
    end

% 5. Compute polynomial fit to central locations
    fitme(color,:) = findedge2(loc(color,:), nlin, npol);
     
% For comparison with linear edge fit
    [fitme1(color,:)] = findedge2(loc(color,:), nlin, 1);
%
    if npol>3
        x = 0: 1: nlin-1;
        y = polyval(fitme(color,:), x);
        y1 = polyval(fitme1(color,:),x);   
        [r2, rmse, merror] = rsquare(y,loc(color,:));
        if rmse>3
            beep
            disp('Warning: possible edge-fitting problem:')
            disp(['mean edge fitting error: ',num2str(merror)]);
            disp(['          r2: ',num2str(r2),' rmse: ',num2str(rmse)]);
        end
    end 
 
    if pflag ==1
        
        x = 0: 1: nlin-1;
        y = polyval(fitme(color,:), x);
        y1 = polyval(fitme1(color,:),x);  
        
        figure;
        image(a(:,:, color)), colormap('gray(256)');
        axis image
        hold on
        np = 8;
         ln = 1:length(loc(color,:));
        plot(loc(color,1:np:end),ln(1:np:end), 'ro','MarkerSize',7), hold on,
        plot(y1,x, 'b--','LineWidth',1);
        plot(y,x, 'r','LineWidth',1);
        title(pname(color,:)), xlabel('edge location'),ylabel('line');
        hold off;
        disp('Edge  location, pixel');
        tdat = zeros(nlin,2);
        tdat(:,1) = (1:nlin)';
        tdat(:,2) = (loc(color, :))';
       
        x = 0: 1: nlin-1;
        y = polyval(fitme(color,:), x);%%
        y1 = polyval(fitme1(color,:),x);%%   
        [r2, rmse, merror] = rsquare(y,loc(color,:));
        disp(['mean error: ',num2str(merror)]);
        disp(['r2: ',num2str(r2),' rmse: ',num2str(rmse)])
        diff = loc(color,:)-y; 
        
        figure
        plot(diff,x,'*');
        xlabel('Residual, pixel'),ylabel('line');
        hold on
        plot([0,0],[0,nlin],'k--')
        axis ij
        axis([-10 10 0 nlin])
        figure
        histogram(diff,15,'Normalization','probability');
         xlabel('Residual, pixel'),ylabel('Frequency, prob.');
        
                    
    end   % pflag ==1

end                                         % End of loop for each color
clear c
summary{1} = ' '; % initialize

midloc = zeros(ncol,1);
summary{1} = 'Edge location, slope'; % initialize
sinfo.edgelab = 'Edge location, slope';

for i=1:ncol   
% Edge angle in degrees from vertical for saving results
% Minus due to slope from vertical in (x,y) not (i,j) coordinates.
   slout(i) = -fitme1(i,end-1); 
   slout(i)= 180*atan(slout(i))/pi;

if rflag==1                       
           slout(i) = slout(i) + 90; 
end

    % Evaluate equation(s) at the middle line as edge location
    midloc(i) = polyval(fitme(i,:), (nlin-1)/2); 
    summary{i+1} = [midloc(i), slout(i)];
    sinfo.edgedat = [midloc(i), slout(i)];
end

if ncol>2
    summary{1} =    'Edge location, slope, misregistration (second record, G, is reference)';
    sinfo.edgelab = 'Edge location, slope, misregistration (second record, G, is reference)';
    misreg = zeros(ncol,1);
    temp11 = zeros(ncol,3);
    for i=1:ncol
        misreg(i) = midloc(i) - midloc(2);
        temp11(i,:) = [midloc(i), slout(i), misreg(i)];
        summary{i+1}=[midloc(i), slout(i), misreg(i)];            
    end
    sinfo.edgedat = temp11;
    clear temp11
    if io == 5 
        disp('Misregistration, with green as reference (R, G, B, Lum) = ');
        for i = 1:ncol
            fprintf('%10.4f\n', misreg(i))
        end
    end  % io ==5
end  % ncol>2

% Limit number of lines to integer(npix*line slope) as per ISO 12233
% Linear fit is used
    nlin1 = round(floor(nlin*abs(fitme1(end,end-1)))/abs(fitme1(end,end-1))); 
    a = a(1:nlin1, :, 1:ncol);           

if npol>3
        disp(['Edge fit order = ',num2str(npol)]);
end

vslope = -fitme1(end,end-1); % Luminence edge angle if computed, linear fit
slope_deg = 180*atan(vslope)/pi;

if rflag==1                       
         slope_deg =slope_deg + 90; 
end
disp(['Edge angle: ',num2str(slope_deg, 3),' degrees'])
if slope_deg < 1
%   beep, warndlg(['High slope warning ',num2str(slope_deg,3),' degrees'], 'Watch it!')
end

delimage = del;

%Correct sampling inverval for sampling normal to edge
delfac = cos(atan(vslope));

del = del*delfac;  % input pixel sampling normal to the edge   
del2 = del/nbin;   % Supersampling interval normal to edge
samp(3) = del2;
if ncol>2
% Next line correction Feb. 2023
    misreg = delfac*misreg; %colour misregistration normal to edge
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ns = length(summary);
summary{ns+1} = [delimage, del2];
sinfo.samp = [delimage del del2 nbin];

nn =   ceil(npix *nbin);
mtf =  zeros(nn, ncol);
nn2 =  floor(nn/2) + 1;

dcorr = fir2fix(nn2, 3);  % dcorr corrects SFR for response of FIR filter

freqlim = 1;
if nbin == 1
    freqlim = 2;  %%%%%%%%%%%%
end
nn2out = round(nn2*freqlim/2);
nfreq = nn/(2*delimage*nn);    % half-sampling frequency

% **************                Large SFR loop for each color record
esf_all = zeros(nn,ncol);  
lsf = zeros(nn,ncol);  

if wflag==0
    disp(['Tukey window    alpha = ',num2str(alpha)])
end

% 6. Form super-sampled edge profile by shifting and binning
for color=1:ncol
    % project and bin data in 4x sampled array
    [esf, status] = project2(a(:,:,color), fitme(color,:), nbin);
    esf = esf(:);
    esf_all(:,color) = esf;


% 7. Compute 1-D derivative of super-sampled edge data

    c = deriv1(esf', 1, nn, fil2); 
    

    % Added 19 April 2017
    if c(1) == 0
       c(1) = c(2);
    elseif c(end) == 0
       c(end) = c(end-1);
    end  
 
    nn = length(c);
    mm = find(c==max(c));
    mm = mean(mm); 
    
%   Shift array so it is centered. Not necessary, since we retain only
%   modulus of the DFT in step 9 (comment next 2 lines to omit).
%     c = cent(c, mm);
    mm = nn/2;
    
    if wflag ~=0
        win = ahamming(nn, mm);
    else
        win = tukey2(nn, alpha, mm);
    end 

% 8. Apply window to edge-derivative (LSF)
    c = win.*c(:);  
%     lsf(:,color)= c';
    
    if pflag ==1
        figure;
        plot(c); hold on,
        xlabel('n'), ylabel('PSF'),title('psf with window');
        hold off;
        disp(' ********* Hit any key to continue*************** ');
        
    end
% 9. Compute normalized modulus of the DFT of the windowed LSF   
%    Transform, scale and correct for FIR filter response
    temp = abs(fft(c, nn));
    mtf(1:nn2, color) = temp(1:nn2)/temp(1);
    
% 10. Correct e-SFR data for discrete derivative response    
    mtf(1:nn2, color) = mtf(1:nn2, color).*dcorr;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end     % color=1:ncol

% 11. Compute colour misregistration, sampling efficiency, spatial
%     frequency, plot and report results

esf = esf(1:nn,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute frequency values
freq = zeros(nn, 1);
for n=1:nn   
    freq(n) = (n-1)/(del2*nn);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dat = zeros(nn2out, ncol+1);
for i=1:nn2out
    dat(i,:) = [freq(i), mtf(i,:)];
end

% Add color misregistration to fitme matrix
temp1 = zeros(ncol, npol+2);
temp1(1:ncol,1:npol+1) = fitme;
if ncol>2
 temp1(:,end) = misreg;
fitme = temp1;
end

ns = length(summary);
summary{ns+1}=fitme; 
sinfo.fitme = fitme;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sampling efficiency
%Values used to report: note lowest (10%) is used for sampling efficiency
val = [0.1, 0.5];

[e, freqval, ~] = sampeff(dat, val, delimage, 0, 0);  %%%%% &&&
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sfr50 = freqval(2);  % frequency where SFR = 50%
ns = length(summary);
summary{ns+1} = e;
summary{ns+2} = freqval;
summary{ns+3} = npol;
summary{ns+4} = samp;

sinfo.se = e;
sinfo.sfr1050 = freqval;
sinfo.npol = npol;

if io ==1         
    return
end
% Plot SFRs on same axes
if ncol >1
  sym{1} = []; 
  sym{1} = '--r';
  sym{2} = '-g';
  sym{3} = '-.b';
  sym{4} = '*k';
  ttext = filename;
%   legg = [{'r'},{'g'},{'b'},{'lum'}];
else
  ttext = filename;
  sym{1} = 'k';
end

pos = round(centerfig(1, 0.6,0.6));
  
%%%%%%%%
figure('Position',pos)
 plot( freq( 1:nn2out), mtf(1:nn2out, 1), sym{1});
 hold on;
  title(ttext, 'interpreter','none');
  xlabel(['     Frequency, ', funit]);
  ylabel('SFR');
	if ncol>1
		for n = 2:ncol-1
			plot( freq( 1:nn2out), mtf(1:nn2out, n), sym{n});
        end
		ndel = round(nn2out/30);
		plot(  freq( 1:ndel:nn2out), mtf(1:ndel:nn2out, ncol), 'ok',...
            freq( 1:nn2out), mtf(1:nn2out, ncol), 'k')
		
            line([nfreq ,nfreq],[.05,0]); 
            h=legend(['r   ',num2str(e(1,1)),'%'],...
                        ['g   ',num2str(e(1,2)),'%'],...
                        ['b   ',num2str(e(1,3)),'%'],...
                        ['L   ',num2str(e(1,4)),'%']);
                    
            pos1 =  get(h,'Position');
            set(h,'Position', [0.97*pos1(1) 0.93*pos1(2) pos1(3) pos1(4)])
            set(get(h,'title'),'String','Sampling Efficiency');             
				
		else % (ncol ==1)
                line([nfreq ,nfreq],[.05,0]);
                h = legend([num2str(e(1)),'%']);
                get(h,'Position');
                pos1 =  get(h,'Position');
                set(h,'Position', [0.97*pos1(1) 0.93*pos1(2) pos1(3) pos1(4)])
                set(get(h,'title'),'String','Sampling Efficiency');          			

    end % ncol>1

   text(.95*nfreq,+.08,'Half-sampling'),

 hold off;
 maxfplot = max(freq(round(0.75*nn2out)), 1.04*nfreq);
 axis([0 maxfplot,0,max(max(mtf(1:nn2out,:)))]);

drawnow
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
defname = [pathname,'*.xls'];
   [outfile,outpath]=uiputfile(defname,'File name to save results (.xls will be added)');
   foutfile=[outpath,outfile];

   if size(foutfile)==[1,2]
      if foutfile==[0,0]
         disp('Saving results: Cancelled')
      end
   else
       
    nn = find(foutfile=='.', 1);
    if isempty(nn) ==1
       foutfile=[foutfile,'.xls'];
    end

    results5(dat, filename, roi, oename, sinfo, foutfile);
   end

% Reset text interpretation
  set(0, 'DefaultTextInterpreter', 'tex')
  cd(home);                % Return to working directory
 
return;

