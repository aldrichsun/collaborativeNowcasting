function [tpv, fpv, tnv, fnv] = evaluation( res, gt, nnowcast, ...
    ACTkf, ACT, visual, nthres, k, outpred, dpdir, PanelName)
% This function evaluates the model with the chosen parameters

thrs = median(ACTkf(ACT==1));
if isnan(thrs)
    thrs = median(ACTkf(ACT==0));
end

% It's a good choice to use median(ACTkf(ACT==1)). However, we still want to
% see the effect of different thresholds and robustness of the chosen thres.
% Therefore, we try a fixed number 'nthres' (e.g., 10) of different thresholds.
tpv = zeros(nthres, 1);
fpv = zeros(nthres, 1);
tnv = zeros(nthres, 1);
fnv = zeros(nthres, 1);

if outpred == 1
    mpred = zeros(nthres, nnowcast);
    fullpred = zeros(nthres, size(ACTkf,1));
end
minthres = median(ACTkf);
maxthres = median(ACTkf(ACT==1));
step = (maxthres-minthres)/(nthres-1);
thrsix = 1;
for thrs = minthres:step:maxthres
    
tp = sum( gt==1.0 & res >= thrs );
fp = sum( gt<1.0 & res >= thrs );
tn = sum( gt<1.0 & res < thrs );
fn = sum( gt==1.0 & res < thrs );

tpv(thrsix) = tp;
fpv(thrsix) = fp;
tnv(thrsix) = tn;
fnv(thrsix) = fn;

if outpred == 1
for prepos = 1:nnowcast
    if( res(prepos,1) >= thrs )
        mpred(thrsix, prepos) = 1;
    end
end
for prepos = 1:size(ACTkf,1)
    if( ACTkf(prepos,1) >= thrs )
        fullpred(thrsix, prepos) = 1;
    end
end
end

thrsix = thrsix + 1;

pos_pre = tp/(tp+fp);
pos_rec = tp/(tp+fn);
neg_pre = tn/(fn+tn);
neg_rec = tn/(fp+tn);

if visual == 1
disp(['Threshold = ', num2str(thrs)]);

disp(['         ||==================================|'])
disp(['         ||             PREDICTED            |'])
disp(['  TRUTH  ||   positive     |     negative    | RECALL'])
disp(['         ||==================================|'])
disp([' positive||    ', num2str(tp), '           |      ', num2str(fn), '          |',num2str(pos_rec),'(',num2str(tp),'/',num2str(tp+fn),')'])
disp([' negative||    ', num2str(fp), '           |      ', num2str(tn), '          |',num2str(neg_rec),'(',num2str(tn),'/',num2str(fp+tn),')'])
disp(['         ||==================================|'])
disp([' PRECISION ',num2str(pos_pre),'(',num2str(tp),'/',num2str(tp+fp),')         ', num2str(neg_pre),'(',num2str(tn),'/',num2str(fn+tn),')'])
end
end
if outpred == 1
    dlmwrite(strcat(dpdir,PanelName{k}), mpred); % write out the prediction result
    dlmwrite(strcat(dpdir,'full_',PanelName{k}), fullpred); % write out the full prediction result
end
if visual == 1
    plot([NaN(sum(~isnan(ACT)),1);gt], 'gx--')
    hold on
    plot(ACT, 'r-')
    plot(ACTkf, 'bx--')
    hold off
end

end
