clear;
warning off backtrace;

%% -------------- Data Source Parameters and Loops ---------------
dat_list = {'intent'};
siz = size(dat_list, 2);
for dat = 1:siz
for tsl = [1]; %[1 2 4]
act_type = dat_list{dat};
timesteplen = num2str(tsl);
personalTime = 'False';
rawdatasrc = strcat(act_type, '\', timesteplen, '_', personalTime, '\');
redatasrc = strcat(act_type, '\', timesteplen, '_', personalTime, '_py\'); % to speed up, we later use python to generate these files
rawdatapath = strcat(pwd, '\dataExample\', rawdatasrc);
redatapath = strcat(pwd, '\dataExample\', redatasrc);
if ~exist(redatapath, 'dir')
	datasrc = rawdatasrc; datapath = rawdatapath; rerun = false;
else
	datasrc = redatasrc; datapath = redatapath; rerun = true;
end
datapattern = strcat(datapath, '*.csv');
files = dir(datapattern);
if( isempty(files) )
    error(strcat('no input files found in ', datapath));
end
% --- Readin all the panels into XX cell array ---
disp('Reading in...');
tic
if ~rerun % running for the first time
XX = cell(length(files), 1);
SeriesName = cell(length(files), 1); % For debug purposes, only valid with the raw data
PanelName = cell(length(files), 1);
for k = 1:length(files) %% TODO: write all the panels in a single file to reduce the IO and avoid xlsread
    UserPanel = strcat(datapath, files(k).name);
    [NUM,TXT] = xlsread(UserPanel);
    NUM = NUM'; %for files generated directly by createPanel.py
    XX{k} = NUM(2:end,2:end);
    SeriesName{k} = TXT(2:end-1,1);
    PanelName{k} = files(k).name;
    [status,cmdout] = system('taskkill /F /IM EXCEL.EXE');
end
else
XX = cell(length(files), 1);
SeriesName = cell(length(files), 1);
PanelName = cell(length(files), 1);
for k = 1:length(files)
    UserPanel = strcat(datapath, files(k).name);
    rtemp = dlmread(UserPanel, ','); % re-readin the panel
    XX{k} = rtemp';
    PanelName{k} = files(k).name;
end
end
ms = round(toc * 1000);
disp(['Elapsed time ', num2str(ms), ' ms']);
if ~rerun % running for the first time
disp('writing out panel files...');
mkdir(redatapath);
for k = 1:length(files)
	reUserPanel = strcat(redatapath, files(k).name); % write out for fast reading in next time
	dlmwrite(reUserPanel, XX{k}');
end
end 
%return
%% ------------------------------------------------------------------

%% -------------- Model Parameters and Loops ---------------
for param_tq = 2:2
for param_tr = 4:4
for param_tp = 1:1
q = param_tq; r = param_tr; p = param_tp;
nnow = 0.25;
visual = 0; %0 false, 1 true % Parameter for printing on screen
outfac = 0; %Switch for outputing factors
outdetail = 0; %Output detailed results for observation and turning. Note: each threshold has one detail file.
outpred = 0; %Output the detailed predicted results
outloading = 0;
featureName = 'test1'; % Differentiate output for vairous manually chosen settings
param_port = strcat('_q=', num2str(q), '_r=', num2str(r), '_p=', num2str(p), '_nnow=', num2str(nnow));
if ~strcmp(featureName, '')
    param_port = strcat(param_port, '_', featureName);
end
%% ----------------------------------------------------------

%% --------------------- Starting Experiments -----------------------------
modelList = {'NowcastParafac2'};
numOfModel = size(modelList, 2);
for mo = 1:numOfModel
modelSwitcher = modelList{mo};
%------ Result dir and file -------
resdir = strcat(fileparts(pwd), '\NowcastingResult\', modelSwitcher, '_result\');
if ~exist(resdir, 'dir')
    mkdir(resdir)
end
respath = strcat(resdir, modelSwitcher, '_', act_type, '_',  timesteplen, '_', personalTime, '_res', param_port, '.csv');
%------ Write factors to file -----
if outfac == 1
    ttdir = strcat(fileparts(pwd), '\NowcastingResult\', modelSwitcher, '_traintest\', act_type, '\now_', timesteplen, '_', personalTime, param_port, '\');
    pnpath = strcat(ttdir, 'pos_neg_instances.txt');
    if ~exist(ttdir, 'dir')
        mkdir(ttdir)
    end
end
%------ Write detailed predicted results to file -----
dpdir = strcat(fileparts(pwd), '\NowcastingResult\', modelSwitcher, '_pred\', act_type, '\now_', timesteplen, '_', personalTime, param_port, '\');
if ~exist(dpdir, 'dir')
    mkdir(dpdir)
end
%-----------------------------------

disp('');
disp('========================================================================');
disp(['Exp: ', modelSwitcher, ' on ', datasrc, ' using params ', param_port]);
tic

% Extract initial estimate of factors using PARAFAC2
if strcmp(modelSwitcher, 'NowcastParafac2')
    disp('CF extracting factors using Parafac2...');
    %Feed the panel tensor to PARAFAC2 model to extract common factors
    [A,H,C,P] = Parafac2Factors(XX, q, r, p, nnow);
    disp('done!');
end
%%------------------------------------------------------------------------
%% Nowcast for each individual user
nthres = 3;
macro_res = zeros(length(files), 4, nthres);
micro_res = zeros(1, 4, nthres);
dres = cell(length(files), 11, nthres); % Here 11 is the number of columns in the detailed result file
for thrix = 1:nthres
dres(1,:,thrix) = [cellstr('anid'),cellstr('pfscore'),cellstr('pos_pre'),cellstr('pos_rec'), ...
    cellstr('nfscore'), cellstr('neg_pre'), cellstr('neg_rec'), ...
    cellstr('tp'), cellstr('fn'), cellstr('fp'), cellstr('tn')];
end
panelCnt = 1; %record the number of compputed users/results, i.e., num. of valid panels
disp('Processing individual user...')
for k = 1:length(files)
    if outfac == 1
        ttpath = strcat(ttdir, num2str(panelCnt-1));
    else
        ttpath = '';
    end
    %----------------------Nowcast for individual user -----------------------%
    %try
        if strcmp(modelSwitcher, 'NowcastParafac2')
            Loading = diag(C(k,:))*(P{k}*H)'; %C_k*(P_k*H*)'
            InitF = A; %A
            [tpv, fpv, tnv, fnv] = NowcastParafac2(XX{k}, q, r, p, nnow, visual, outfac, ...
                ttpath, nthres, Loading, InitF, k, outpred, dpdir, PanelName);
        end
    %catch ME
    %    warning(strcat(modelSwitcher, ': Error occurs while processing the panel'));
    %    continue;
    %end
    %--------------------------------------------------------------------------------------------------%
    %% Loop through the result vector coming from different thresholds
    for thrix = 1:nthres
        tp = tpv(thrix); fp = fpv(thrix); tn = tnv(thrix); fn = fnv(thrix);
        pnnum = [panelCnt-1, tp+fn, fp+tn];
        if outfac == 1 && thrix == 1 % write out the number of positive and negative instances only once
            if panelCnt-1 == 0
                dlmwrite(pnpath, pnnum, 'delimiter', '\t');
            else
                dlmwrite(pnpath, pnnum, 'delimiter', '\t', '-append');
            end
        end
        % compute result: macro/micro precision and recall
        pos_pre = tp/(tp+fp);
        if isnan(pos_pre)
            pos_pre = 0;
        end
        pos_rec = tp/(tp+fn);
        if isnan(pos_rec)
            pos_rec = 0;
        end 
        neg_pre = tn/(fn+tn);
        if isnan(neg_pre)            
            neg_pre = 0;
        end
        neg_rec = tn/(fp+tn);
        if isnan(neg_rec)
            neg_rec = 0;
        end
        macro_res(panelCnt,:, thrix) = [pos_pre,pos_rec,neg_pre,neg_rec];
        micro_res(:,:,thrix) = micro_res(:,:,thrix) + [tp, fp, tn, fn];
        pfscore = 2*(pos_pre*pos_rec)/(pos_pre+pos_rec);
        if isnan(pfscore)
            pfscore = 0;
        end
        nfscore = 2*(neg_pre*neg_rec)/(neg_pre+neg_rec);
        if isnan(nfscore)
            nfscore = 0;
        end
        dres(panelCnt+1,:,thrix) = [cellstr(files(k).name), pfscore, pos_pre, pos_rec, ...
            nfscore, neg_pre, neg_rec, tp, fn, fp, tn];
    end % end loop through threshold result vector
    panelCnt = panelCnt + 1;
end
ms = round(toc * 1000);
disp(['Elapsed time ', num2str(ms), ' ms']);

for thrix = 1:nthres
maave = mean(macro_res(:,:,thrix));
mapfscore = 2*(maave(1)*maave(2)/(maave(1)+maave(2)));
maave(5) = mapfscore;
mapnscore = 2*(maave(3)*maave(4)/(maave(3)+maave(4)));
maave(6) = mapnscore;

mipp = micro_res(:,1,thrix)/(micro_res(:,1,thrix)+micro_res(:,2,thrix));
mipr = micro_res(:,1,thrix)/(micro_res(:,1,thrix)+micro_res(:,4,thrix));
minp = micro_res(:,3,thrix)/(micro_res(:,4,thrix)+micro_res(:,3,thrix));
minr = micro_res(:,3,thrix)/(micro_res(:,2,thrix)+micro_res(:,3,thrix));
miave = [mipp, mipr, minp, minr];
mipfscore = 2*(miave(1)*miave(2)/(miave(1)+miave(2)));
miave(5) = mipfscore;
minfscore = 2*(miave(3)*miave(4)/(miave(3)+miave(4)));
miave(6) = minfscore;

%write result to file
if ~exist(resdir, 'dir')
    mkdir(resdir)
end
if thrix == 1
    dlmwrite(respath, maave);
else
    dlmwrite(respath, maave, '-append');
end
dlmwrite(respath, miave, '-append');

tunpath = strcat(resdir, 'detail', param_port, '_thr=', num2str(thrix), '.xls');
xlswrite(tunpath, dres(:,:,thrix));
end
disp(['Write results into ', respath, ' successfully']);
end
end
end
end
end
end
disp('========================================================================');
