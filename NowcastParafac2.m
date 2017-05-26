function [tpv, fpv, tnv, fnv] = NowcastParafac2(DATA,q,r,p,nnow,visual,outfac,outpartialdir,nthres,Loading,InitF, ...
    k, outpred, dpdir, PanelName);

[x, nnow, gt] = preprocessPanel(DATA, q, r, p, nnow);
[T,N] = size(x);
nnowcast = floor(T*nnow);

[F,VF,A,C] = IndiviFactor(x(:,1:end-1),q,r,p,nnow,Loading,InitF);

ACT = x(1:end,end);

if outfac == 1 % output factors
    FTT = [DATA(1:end,end) F];
    training = FTT(1:end-nnowcast, :);
    testing = FTT(end-nnowcast+1:end, :);
    dlmwrite(strcat(outpartialdir, '_training.txt'), training, 'delimiter', '\t');
    dlmwrite(strcat(outpartialdir, '_testing.txt'), testing, 'delimiter', '\t');
end

Fq = F(1:end,:);

% Find out the last available data point for ACT regression
temp = sum(isnan(ACT)==0); % Here 'temp' means contemporaneous

% Bridge regression
Z = [ones(size(Fq(:,1))) Fq]; % Regressors
%To-do: reorganize the code
%Z = Z(1:temp,:)
%ACT_backup = ACT(:,:)
%ACT = ACT(1:temp,:)
beta = inv(Z(1:temp,:)'*Z(1:temp,:))*Z(1:temp,:)'*ACT(1:temp,:);%% Regression coefficients
%beta = inv(Z'Z)Z'*ACT
Vidio = var(Z(1:temp,:)*inv(Z(1:temp,:)'*Z(1:temp,:))*Z(1:temp,:)'*ACT(1:temp,:) - ACT(1:temp,:));%% Residual variance
%Vidio = var(Z*beta - ACT)
ACTkf = Z*inv(Z(1:temp,:)'*Z(1:temp,:))*Z(1:temp,:)'*ACT(1:temp,:);%% Fit
%ACTkf = Z*beta

% -----------------------------
% Factor filter uncertainty
for jt = 1:size(VF,3)
    Vchi(jt,:) = beta(2:end)'*VF(1:p*r,1:p*r,jt)*beta(2:end);
end;
VqChi = Vchi(1:end,:);

Vact = zeros(size(ACT)); Vact(isnan(ACT)) = VqChi(isnan(ACT))+Vidio;
STDpred = sqrt(Vact);
% -----------------------------

ACTpred = ACT;
ACTpred(isnan(ACT))=ACTkf(isnan(ACT)); % Predicts ACT

res = ACTpred(temp+1:temp+nnowcast,:); % ACTpred = ACT(traning) + ACTkf(testing)

% Evaluate the model with the chosen parameters
[tpv, fpv, tnv, fnv] = evaluation( res, gt, nnowcast, ...
    ACTkf, ACT, visual, nthres, k, outpred, dpdir, PanelName);
end
