function [F,VF,A,C,Q,R,initx,initV,ss,MM] = IndiviFactor(x,q,r,p,nnow,inputC,initF);
% Estimate a posteriori factors from vectors of time series that are possibly unbalanced
% at the end, (NaN for missing observations).
%
% Code based on: "Nowcasting: The Real Time Informational Content of Macroeconomic Data",
% Domenico Giannone, Lucrezia Reichlin & David Small, Journal of Monetary Economics.
%
% The model (and output):
%   x_t = C F_t + \xi_t
%   F_t = A F_{t-1} + B u_t
%   u_t ~ WN(0,I_q)
%   R = E(\xi_t \xi_t')
%   Q = BB'
%   initx = F_0
%   initV = E(F_0 F_0')
%   q: dynamic rank
%   r: static rank (r>=q)
%   p: auto-regressive order of the state vector (default p=1)
%   F: estimated factors
%   VF: estimation covariance of factors
%   ss: std(x) 
%   MM: mean(x)
%
% To obtain the a posteriori factors, we need first estimate parameters in the model.
% From Parafac2, we have obtained C and F_t (in matrix form). We need first compute A, Q and R.

[T,N] = size(x); %% dimension of the panel

% First construct the balanced panel z from the original panel x, and estimate the parameters A, Q and R 
% by regression on initial estimates of the factors (based on the balanced part of the panel).
z = balancePanel(x,q,r,p,nnow);

% Standardize the panel
ss = std(z);	% computes stdev of each column of data.
MM = mean(z);
s = ones(T,1)*ss; % Copy the stdev vector into T rows.
M = ones(T,1)*MM;
x = (x - M)./s;

%z = x(1:end-nnowcast, :);
z = balancePanel(x,q,r,p,nnow);

% First, estimate A, Q and R.
[A, C, Q, R, initx, initV] = paramEstimator(z, q, r, p, inputC, initF); % z stands for the observation for parameter estimation

% And then smooth the factors with real observations (that may contain missing values).
%
% The estimatation of a posteriori factors in presence of missing data is 
% performed by using (time varying) Kalman filter in which missing data are
% assigned an extremely large noise variance.

%% Prepare the parameters of the (time varying) state space model. Time is the 3rd dimension.
for t = 1:T
    % We assume that the transition matrix A, loading matrix C, system 
    % covariance Q, and measurement covariance R are time-invariant.
    AA(:,:,t) = A;
    CC(:,:,t) = C;
    QQ(:,:,t) = Q;
    % Handle missing values
    miss = isnan(x(t,:));
    Rtemp = diag(R); % Here input is a matrix, diag(R) returns a (column) vector of the main diagonal elements of R.
    % Missing data are assigned an extremely large measurement variance
    Rtemp(miss)=1e+32;
    RR(:,:,t) = diag(Rtemp); % Input is a vector, diag(Rtemp) returns a square diagonal matrix with the elements of Rtemp on the main diagonal
end;

% Copy observations and assign missing data an arbitrary value
y = x; y(isnan(x))=0;
y = y'; % to be compatible with the kalman smoother function

%% Run the kalman smoother on the (time varying) state space model
[xsmooth, Vsmooth, VVsmooth, loglik] = kalman_smoother_diag(y, AA, CC, QQ, RR, initx, initV, 'model', 1:T);
% The input is
%   y - observation
%   A - transition matrix
%   C - measurement (loading/observation) matrix
%   Q - system (transition) covariacne
%   R - measurement error covariance (diagonal with missing value having a
%       very large variance)
%   initx - the initial state (i.e., the first factor f_1 = F(1,:)))
%   initV - the initial state covariance (i.e., cov(initx))
%   The input arguments with double letters (e.g., AA) means tensors consisting of T copies of 
%   the corresponding matrix named by the single letter.
%
% The output is
%   xsmooth = E[F_t|y(:,1:T)] the a posteriori states (factors) f_t = F(t,:), i.e., filtered (smoothed) factors
%   Vsmooth = Cov[F_t|y(:,1:T)] the a posteriori error covariance
%   VVsmooth = Cov[F_t+1,F_t|y(:,1:T)] the a posteriori cross-step error covariance
%   loglik = sum{t=1}^T log P(y(:,t)) the sum of the log-likelihood of observations

% After the filtering/smoothing step, we obtain the final estimated factors (and other quantities)
% xsmooth = E(F_t|y(:,1:T))
% Vsmooth = VAR(F_t|y(:,1:T))
F =  xsmooth';
VF = Vsmooth;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A, C, Q, R, initx, initV, Mx, Wx] = paramEstimator(x, q, r, p, inputC, initF)
% This function computes the parameters of the factor model
% Note that C and R refer to the standardized variables.
% The meanings of all returned variables are:
%   A - the system matrix, i.e., factor trasition matrix
%   C - the observation matrix, i.e., the measurement matrix
%   Q - the system covariance, i.e., the trasition covariance
%   R - the observation covariance, i.e., the measurement covariance
%   initx - the initial state (column) vector 
%   initV - the initial state covariance 
%   Mx - the mean row vector of input x, which here contains only zeros.
%   Wx - the standard deviation row vector of input x, which contains 1.

% The standardize procedure
Mx = mean(x); % Mean: the (row) vector contains only zeros
Wx = diag(std(x)); % Standard deviation: the stds are 1.0
x = center(x)*inv(Wx); % Standardize. Here, center is a sub-function, which is claimed to be faster than the mean method.

OPTS.disp = 0;
[T,N] = size(x);
if r < q; % static rank r cannot be larger than the dynamic rank
    error('q has to be less or equal to r')
end

% Define preliminary quantities to write VAR in the companion form 
A_temp = zeros(r,r*p)';	% a zero matrix,
I = eye(r*p,r*p);		% identity matrix
A = [A_temp';I(1:end-r,1:end)]; % If p=1, I(1:end-r,1:end) is empty, which makes A = A_temp.
Q = zeros(r*p,r*p);	% a zero matrix
Q(1:r,1:r) = eye(r); % identity of size r

% Estimation of parameters
F = initF; % The factors estimated by PARAFAC2

% Estimate the measurement covariance matrix, where (x - inputC*F) is 
% the residuals between measurement/observation and transition computation.
R = diag(diag(cov(x - F*inputC)));

% Estimate the autoregressive model for the factors: run VAR F(t) = A_1*F(t-1)+...+A_p*F(t-1) + e(t);
if p > 0
    % We first use OLS estimator to estimate the VAR transition matrix A
    % ------------------------------------------------------------------
    % In what follows, for p=1, we have 
    %       z = [F_2, F_3, ..., F_T]',
    %       Z = [F_1, F_2, ..., F_{T-1}]', and
    %       z = ZA' + error_term, i.e., z' = AZ'.
    %
    % According to wiki (https://en.wikipedia.org/wiki/Vector_autoregression)
    % when 
    %       Y = BZ + U,
    % the estimate of 
    %       B = YZ'(ZZ')^{-1}.
    % Here we have 
    %       Y == z', Z (in formula) == Z' (in code), and B == A. 
    %
    % Therefore, 
    %       A = z'Z(Z'Z)^{-1}, which in code is A = z'*Z*inv(Z'*Z).
    % ------------------------------------------------------------------
    z = F;
    Z = [];
    for kk = 1:p
        Z = [Z z(p-kk+1:end-kk,:)]; % stacked regressors
    end;
    z = z(p+1:end,:);
    A_temp = inv(Z'*Z)*Z'*z; % OLS estimator of the VAR transition matrix
    A(1:r,1:r*p) = A_temp';
    
    % Estimate Q, i.e., the trasition covariance matrix
    e = z  - Z*A_temp; % VAR residuals
    H = cov(e); % VAR covariance matrix

    if r == q % The covariance matrix of VAR residuals is of full rank
        Q(1:r,1:r) = H;
    else % The covariance matrix of VAR residuals has reduced rank
        [P,M] = eigs(H,q,'lm',OPTS);  % eigenvalue decomposition
        P = P*diag(sign(P(1,:)));
        Q(1:r,1:r) = P*M*P';
    end;
end;

% Estimate the initial conditions for Kalman Filter.
nlag = p-1;	% p=1, so nlag = 0.
% Initial variance is set equal to the unconditional variance of the factors.
if p > 0
    z = F;
    Z = [];
    for kk = 0:nlag % When p = 1, nlag = 0. Therefore, kk = 0 and Z (= z(1:end,:) = z) = F;
        Z = [Z z(nlag-kk+1:end-kk,:)]; % stacked regressors
    end;
    initx = Z(1,:)'; % Z = F 
    initV = reshape(pinv(eye(size(kron(A,A),1))-kron(A,A))*Q(:), r*p, r*p);
	%This is the initial covariance of the factors in a more general form. Normally, we have initV = cov(initx).
else 
    initx = [];
    initV = [];
end;

% When p = 1, we have nlag = 0. So we simply transpose the input measurement
% matrix (estimated by PARAFAC2) to be compatible with other parts.
C = [inputC' zeros(N,r*(nlag))];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xsmooth, Vsmooth, VVsmooth, loglik] = kalman_smoother_diag(y, A, C, Q, R, init_x, init_V, varargin)
% Adapted from programs by Zoubin Ghahramani and Geoffrey E. Hinton, priviously available at http://www.gatsby.ucl.ac.uk/~zoubin, 1996.
% Kalman/RTS smoother.
% [xsmooth, Vsmooth, VVsmooth, loglik] = kalman_smoother_diag(y, A, C, Q, R, init_x, init_V, ...)
%
% The inputs are the same as for kalman_filter.
% The outputs are almost the same, except we condition on y(:,1:T) (and u(:,1:T) if specified),
% instead of on y(:, 1:t). Here y stands for the observation.
%
% This function is named '_diag' because it assumes that R (the measurement covariance)
% is diagonal (which means the measurement errors at different time steps are independent),
% and this enables more efficient computation the Kalman gain (without costly matrix inverse).
%
% INPUT:
%   y - observation
%   A - transition matrix
%   C - measurement (loading/observation) matrix
%   Q - system (transition) covariacne
%   R - measurement error covariance (diagonal with missing value having a
%       very large variance)
%   init_x - the initial state (i.e., the first factor f_1 = F(1,:)))
%   init_V - the initial state covariance (i.e., cov(initx))
%   varargin - (varied argument input) are OPTIONAL INPUTS (string/value pairs [default in brackets])
%        'model' - indicates the order/sequence of observations/factors
%        'u'     - u(:,t) the control signal at time t
%        'B'     - B(:,:,t) the control signal matrix for model m
%   Note that the input arguments are tensors consisting of T copies of corresponding matrices.
%
% OUTPUT:
%   xsmooth = E[F_t|y(:,1:T)] the a posteriori states (factors) f_t = F(t,:), i.e., (RTS) smoothed (filtered) factors
%   Vsmooth = Cov[F_t|y(:,1:T)] the a posteriori error covariance
%   VVsmooth = Cov[F_t+1,F_t|y(:,1:T)] the a posteriori cross-step error covariance
%   loglik = sum{t=1}^T log P(y(:,t)) the sum of the log-likelihood of observations

[os T] = size(y);
ss = size(A,1);

% set default params
model = ones(1,T);
u = [];
B = [];

args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i}
        case 'model', model = args{i+1};
        case 'u', u = args{i+1};
        case 'B', B = args{i+1};
        otherwise, error(['unrecognized argument ' args{i}])
    end
end

xsmooth = zeros(ss, T);
Vsmooth = zeros(ss, ss, T);
VVsmooth = zeros(ss, ss, T);

% Forward pass
[xfilt, Vfilt, VVfilt, loglik] = kalman_filter_diag(y, A, C, Q, R, init_x, init_V, ...
    'model', model, 'u', u, 'B', B);
% ----------------------------------------------------------------------
% The meanings of input are as follows.
%   y - the observation,
%   A - the system/transition matrix,
%   C - the measurement matrix,
%   Q - the system noise covariance matrix, 
%   R - the measurement noise covariance matrix, 
%   init_x - the estimated initial state (factor) of the system, 
%   init_V - the estimated initial error covariance,
%   u - the extraneous control force and 
%   B - the control matrix.
%
% The output:
%   xfilt - the a posteriori state (factor) x(:,t), i.e., filtered result and
%   Vfilt - the a posteriori error covariance is V(:,:,t)
%   VVfilt - Cov[ x(t), x(t-1) | y(:, 1:t) ], is the cross-covariance (which assumed to be zero in our problem)
%   loglik - the sum of the log-likelihood of the observations (which is not used in our problem).
%       Note that the 'filt' means filtered results.
% ----------------------------------------------------------------------

% Backward pass
xsmooth(:,T) = xfilt(:,T);
Vsmooth(:,:,T) = Vfilt(:,:,T);
VVsmooth(:,:,T) = VVfilt(:,:,T);

for t=T-1:-1:1
    m = model(t+1);
    if isempty(B) % In our current design rationale, B is not used, i.e., empty
        [xsmooth(:,t), Vsmooth(:,:,t), VVsmooth(:,:,t+1)] = ...
            smooth_update(xsmooth(:,t+1), Vsmooth(:,:,t+1), xfilt(:,t), Vfilt(:,:,t), ...
            Vfilt(:,:,t+1), VVfilt(:,:,t+1), A(:,:,m), Q(:,:,m), [], []);
        % ----------------------------------------------------------------------
        % INPUTS:
        %   xsmooth(:,t+1) - xsmooth_future = E[X_t+1|T]
        %   Vsmooth(:,:,t+1) - Vsmooth_future = Cov[X_t+1|T] (the abvoe two are like observations in Kalman filter)
        %   xfilt(:,t) - xfilt = E[X_t|t]
        %   Vfilt(:,:,t) - Vfilt = Cov[X_t|t]
        %   Vfilt(:,:,t+1) - Vfilt_future = Cov[X_t+1|t+1]
        %   VVfilt(:,:,t+1) - VVfilt_future = Cov[X_t+1,X_t|t+1]
        %   A(:,:,t+1) - A = system matrix for time t+1
        %   Q(:,:,t+1) - Q = system covariance for time t+1
        %   [] - B = input matrix for time t+1 (or [] if none)
        %   [] - u = input vector for time t+1 (or [] if none)
        %
        % OUTPUTS:
        %   xsmooth(:,t) - xsmooth = E[X_t|T]
        %   Vsmooth(:,:,t) - Vsmooth = Cov[X_t|T]
        %   VVsmooth(:,:,t+1) - VVsmooth_future = Cov[X_t+1,X_t|T]
        % ----------------------------------------------------------------------
    else
        % --------------- currently unreachable code -------------------
        [xsmooth(:,t), Vsmooth(:,:,t), VVsmooth(:,:,t+1)] = ...
            smooth_update(xsmooth(:,t+1), Vsmooth(:,:,t+1), xfilt(:,t), Vfilt(:,:,t), ...
            Vfilt(:,:,t+1), VVfilt(:,:,t+1), A(:,:,m), Q(:,:,m), B(:,:,m), u(:,t+1));
        % --------------------------------------------------------------
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, V, VV, loglik] = kalman_filter_diag(y, A, C, Q, R, init_x, init_V, varargin)
% Adapted from programs by Zoubin Ghahramani and Geoffrey E. Hinton, previously available at http://www.gatsby.ucl.ac.uk/~zoubin, 1996.
% Kalman filter.
% [x, V, VV, loglik] = kalman_filter_diag(y, A, C, Q, R, init_x, init_V, ...)
%
% INPUTS:
%   y(:,t) - the observation at time t
%   A - the system matrix
%   C - the observation matrix 
%   Q - the system covariance 
%   R - the observation covariance
%   init_x - the initial state (column) vector 
%   init_V - the initial state covariance 
%
% OPTIONAL INPUTS (string/value pairs [default in brackets])
% 'model' - model(t)=m means use params from model m at time t [ones(1,T)]
%     In this case, all the above matrices take an additional final dimension,
%     i.e., A(:,:,m), C(:,:,m), Q(:,:,m), R(:,:,m).
%     However, init_x and init_V are independent of model(1).
% 'u'     - u(:,t) the control signal at time t [ [] ]
% 'B'     - B(:,:,t) the control signal matrix for model m
%
% OUTPUTS (where X is the hidden state being estimated)
%   x(:,t) = E[X(:,t) | y(:,1:t)]
%   V(:,:,t) = Cov[X(:,t) | y(:,1:t)]
%   VV(:,:,t) = Cov[X(:,t), X(:,t-1) | y(:,1:t)] t >= 2
%   loglik = sum{t=1}^T log P(y(:,t))
%
% If an input (control) signal is specified, we also condition on it:
% e.g., x(:,t) = E[X(:,t) | y(:,1:t), u(:, 1:t)]
% If a model sequence is specified, we also condition on it:
% e.g., x(:,t) = E[X(:,t) | y(:,1:t), u(:, 1:t), m(1:t)]

[os T] = size(y);
ss = size(A,1); % size of state space, i.e., ss = r

% set default params
model = ones(1,T);
u = [];
B = [];
ndx = [];

args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i}
        case 'model', model = args{i+1};
        case 'u', u = args{i+1};
        case 'B', B = args{i+1};
        case 'ndx', ndx = args{i+1}; % related to extraneous control signal
        otherwise, error(['unrecognized argument ' args{i}])
    end
end

x = zeros(ss, T);
V = zeros(ss, ss, T);
VV = zeros(ss, ss, T);

loglik = 0;
for t=1:T
    m = model(t); % we use the normal sequence model = 1:T so t = model(t)
    if t==1
        %prevx = init_x(:,m);
        %prevV = init_V(:,:,m);
        prevx = init_x;
        prevV = init_V;
        initial = 1;
    else
        %prevx = init_x(:,model(t-1));
        %prevV = init_V(:,:,model(t-1));
        prevx = x(:,t-1);
        prevV = V(:,:,t-1);
        initial = 0;
    end
    if isempty(u) % In our current design rationale, u (extraneous control) is not used, i.e., empty
        [x(:,t), V(:,:,t), LL, VV(:,:,t)] = ...
            kalman_update_diag(A(:,:,m), C(:,:,m), Q(:,:,m), R(:,:,m), y(:,t), prevx, prevV, 'initial', initial);
            % This function returns the a posteriori state estimate x(:,t) 
            % and the a posteriori estimate covariance V(:,:,t). LL is the 
            % log-likelihood for y(:,t) and VV is Cov[X(t),X(t-1)|y(:,1:t)].
    else
        % --------------- currently unreachable code -------------------
        if isempty(ndx)
            [x(:,t), V(:,:,t), LL, VV(:,:,t)] = ...
                kalman_update_diag(A(:,:,m), C(:,:,m), Q(:,:,m), R(:,:,m), y(:,t), prevx, prevV, ... 
                'initial', initial, 'u', u(:,t), 'B', B(:,:,m));
        else
            i = ndx{t};
            % copy over all elements; only some will get updated
            x(:,t) = prevx;
            prevP = inv(prevV);
            prevPsmall = prevP(i,i);
            prevVsmall = inv(prevPsmall);
            [x(i,t), smallV, LL, VV(i,i,t)] = ...
                kalman_update_diag(A(i,i,m), C(:,i,m), Q(i,i,m), R(:,:,m), y(:,t), prevx(i), prevVsmall, ...
                'initial', initial, 'u', u(:,t), 'B', B(i,:,m));
            smallP = inv(smallV);
            prevP(i,i) = smallP;
            V(:,:,t) = inv(prevP);
        end
        % --------------------------------------------------------------
    end
    loglik = loglik + LL;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xnew, Vnew, loglik, VVnew] = kalman_update_diag(A, C, Q, R, y, x, V, varargin)
% Adapted from programs by Zoubin Ghahramani and Geoffrey E. Hinton, available at http://www.gatsby.ucl.ac.uk/~zoubin, 1996.
% This function performs a one-step update of the Kalman filter
%
%   [xnew, Vnew, loglik] = kalman_update_diag(A, C, Q, R, y, x, V, ...)
%   This function is named '_diag' because it assumes that R (the measurement covariance)
%   is diagonal (which means the measurement errors at different time steps are independent),
%   and this enables more efficient computation the Kalman gain (without costly matrix inverse).
%
% INPUTS:
%   A - the system matrix
%   C - the measurement (loading) matrix 
%   Q - the system covariance 
%   R - the observation covariance
%   y(:) - the observation at time t
%   x(:) - E[X|y(:,1:t-1)] prior mean
%   V(:,:) - Cov[X|y(:,1:t-1)] prior covariance
%
% OPTIONAL INPUTS (string/value pairs [default in brackets])
%   'initial' - 1 means x and V are taken as initial conditions (so A and Q are ignored)
%   'u'       - u(:) the control signal at time t
%   'B'       - the input control signal matrix
%
% OUTPUTS (where X is the hidden state being estimated)
%   xnew(:) =   E[X|y(:,1:t)] 
%   Vnew(:,:) = Cov[X(t)|y(:,1:t)]
%   VVnew(:,:) = Cov[X(t),X(t-1)|y(:,1:t)]
%   loglik = log P(y(:,t)|y(:,1:t-1)) log-likelihood of observation/innovation
%
% Formula of Kalman filter update:
%   x_t = A*x_t-1
%   V_t = A*V_t-1*A' + Q
%   K_t = V_t*C'*(C*V_t*C' + R)^{-1}
%   xnew_t = x_t + K_t*(y_t - C*x_t)
%   Vnew_t = (I - K_t*C)*V_t

% set default params
u = [];
B = [];
initial = 0;

args = varargin;
for i=1:2:length(args)
    switch args{i}
        case 'u', u = args{i+1};
        case 'B', B = args{i+1};
        case 'initial', initial = args{i+1};
        otherwise, error(['unrecognized argument ' args{i}])
    end
end

% xpred(:) = E[X_t+1|y(:,1:t)]
% Vpred(:,:) = Cov[X_t+1|y(:,1:t)]
% Here 'pred' indicates predicted from system state/factor trasition

if initial
    if isempty(u)
        xpred = x;
    else
        xpred = x + B*u;
    end
    Vpred = V;
else
    if isempty(u)
        xpred = A*x;                    % x_t = A*x_t-1
    else
        xpred = A*x + B*u;
    end
    Vpred = A*V*A' + Q;                 % V_t = A*V_t-1*A' + Q
end

e = y - C*xpred; % error (innovation)   % y_t - C*x_t
n = length(e);
ss = length(A);
d = size(e,1);

S = C*Vpred*C' + R;                     % C*V_t*C' + R
GG = C'*diag(1./diag(R))*C;             % to compute (C*V_t*C' + R)^{-1} and 
% Sinv = inv(S);                        % (C*V_t*C' + R)^{-1} % simplest yet most inefficient method
Sinv = diag(1./diag(R)) - (diag(1./diag(R))*C*pinv(eye(ss)+Vpred*GG)*Vpred*C'*diag(1./diag(R))); % works only with R diagonal

%----------------------
% To compute the log-likelihood.
detS = prod(diag(R))*det(eye(ss)+Vpred*GG);
denom = (2*pi)^(d/2)*sqrt(abs(detS));
mahal = sum(e'*Sinv*e,2);
loglik = -0.5*mahal - log(denom);
%----------------------

K = Vpred*C'*Sinv;                      % V_t*C'*(C*V_t*C' + R)^{-1} % Kalman gain
% If there is no observation vector, set K = zeros(ss).

xnew = xpred + K*e;                     % xnew_t = x_t + K_t*(y_t - C*x_t), e = y_t - C*x_t
Vnew = (eye(ss) - K*C)*Vpred;           % Vnew_t = (I - K_t*C)*V_t
VVnew = (eye(ss) - K*C)*A*V;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The call of this function is: 
function [xsmooth, Vsmooth, VVsmooth_future] = smooth_update(xsmooth_future, Vsmooth_future, ...
    xfilt, Vfilt, Vfilt_future, VVfilt_future, A, Q, B, u)
% Adapted from programs by Zoubin Ghahramani and Geoffrey E. Hinton, previously available at http://www.gatsby.ucl.ac.uk/~zoubin, 1996.
% This function performs one step of the backwards RTS smoothing.
%
% function [xsmooth, Vsmooth, VVsmooth_future] = smooth_update(xsmooth_future, Vsmooth_future, ...
%    xfilt, Vfilt,  Vfilt_future, VVfilt_future, A, B, u)
%
% INPUTS:
%   xsmooth_future = E[X_t+1|T],            i.e., xsmooth(:,t+1) in function 'kalman_smoother_diag'.
%   Vsmooth_future = Cov[X_t+1|T],          i.e., Vsmooth(:,:,t+1) (the abvoe two are like observations in Kalman filter)
%   xfilt = E[X_t|t],                       i.e., xfilt(:,t)
%   Vfilt = Cov[X_t|t],                     i.e., Vfilt(:,:,t)
%   Vfilt_future = Cov[X_t+1|t+1],          i.e., Vfilt(:,:,t+1)
%   VVfilt_future = Cov[X_t+1,X_t|t+1],     i.e., VVfilt(:,:,t+1)
%   A = system matrix for time t+1,         i.e., A(:,:,t+1)
%   Q = system covariance for time t+1,     i.e., Q(:,:,t+1)
%   B = input matrix for time t+1 (or [] if none), i.e., []
%   u = input vector for time t+1 (or [] if none), i.e., []
%
% OUTPUTS:
%   xsmooth = E[X_t|T]
%   Vsmooth = Cov[X_t|T]
%   VVsmooth_future = Cov[X_t+1,X_t|T]
%
% Formula of RTS smoother update:
%   x_t+1|t = A*x_t|t
%   V_t+1|t = A*V_t|t*A' + Q
%   J_t = V_t|t * A' * V_t+1|t^{-1} % smoother gain
%   x_t|T = x_t|t + J_t*( x_t+1|T - x_t+1|t )
%   V_t|T = V_t|t + J_t*( V_t+1|T - V_t+1|t )*J_t'

%xpred = E[X(t+1) | t]
if isempty(B)
    xpred = A*xfilt;
else
    xpred = A*xfilt + B*u;
end
Vpred = A*Vfilt*A' + Q; % Vpred = Cov[X(t+1)|t]
J = Vfilt * A' * pinv(Vpred); % smoother gain matrix
xsmooth = xfilt + J*(xsmooth_future - xpred);
Vsmooth = Vfilt + J*(Vsmooth_future - Vpred)*J';
VVsmooth_future = VVfilt_future + (Vsmooth_future - Vfilt_future)*pinv(Vfilt_future)*VVfilt_future;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function XC = center(X)
% CENTER XC = center(X)
%	Centers each column of X.
%	J. Rodrigues 26/IV/97, jrodrig@ulb.ac.be

[T n] = size(X);
XC = X - ones(T,1)*(sum(X)/T); % Much faster than MEAN with a FOR loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%