function [z] = balancePanel(x,q,r,p,nnow);
% This function determines the balanced panel for estimating initial
% factors initF, loadings C, and transition matrix A, system covariance Q,
% and measurement convariance R

[T,N] = size(x); % dimension of the panel

% ---------------------- Approach one --------------------------
% We use all the historical (training) data to build a balanced panel.
% Since all the missing/outlier values have been properly addressed in
% historical data by the preprocessPanel procedure, here we directly 
% return the historical part.
nnowcast = floor(T*nnow);
z = x(1:end-nnowcast, :);
% --------------------------------------------------------------

% ---------------------- Approach two --------------------------
% Another approach: z is the matrix with the most recent balanced panel 
% (excluding ACT). However, this may results in too few training data 
% when there are many scattered missing values in the panel. If all missing
% values (NaN) are at the end of the panel, then we could deploy
%das = sum(isnan(x));
%m = max(das);
%z = x(1 : T - m , : );
% --------------------------------------------------------------

% ---------------------- Approach three -------------------------
% When the missing values in historical data are not manually processed, 
% we could instead SELECT out a balanced panel to perform the initial 
% estimation, as it exploits more available data, although this might 
% deteriorates the time series to be mix-frequency ones. Later, we may 
% consider MIDAS.
% The following code works when the last column (end) is the ONLY column 
% that could contain NaN in the data sets.
%sel = ~isnan(x(1:end, end));
%z = x(sel, :);
% --------------------------------------------------------------