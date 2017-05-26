function [A,H,C,P] = Parafac2Factors(DATA,q,r,p,nnow);

NumOfUser = size(DATA, 1);
X = cell(NumOfUser, 1);
for k = 1:NumOfUser
    
    x = preprocessPanel(DATA{k},q,r,p,nnow); % To keep consistent with the preprocessing in NowcastParafac2
    
    x = x(:,1:end-1); %exclude the label column
    
    [T,N] = size(x); %% dimension of the (original) panel
    z = balancePanel(x,q,r,p,nnow);
    
    % Standardize the panel and whitens the data (with the balanced part 
    % which produces roughly gaussian numbers as information)
    ss = std(z);	% computes stdev of each column of data.
    MM = mean(z);
    s = ones(T,1)*ss; % Copy the stdev vector into T rows.
    M = ones(T,1)*MM;
    x = (x - M)./s;

    z = balancePanel(x,q,r,p,nnow); % Obtains the balanced part of the standardized panel
    x = z;
    %---------------------- Approach two --------------------------
    
    X{k} = x;
end

% Constraints
%   Vector of length 2. The first element defines constraints
%   imposed in the first mode, the second defines contraints in
%   third mode (the second mode is not included because constraints
%   are not easily imposed in this mode)
% 
%   If Constraints = [a b], the following holds. If 
%   a = 0 => no constraints in the first mode
%   a = 1 => nonnegativity in the first mode
%   a = 2 => orthogonality in the first mode
%   a = 3 => unimodality (and nonnegativity) in the first mode
%   same holds for b for the third mode
%
% Options
%   An optional vector of length 3
%   Options(1) Convergence criterion
%            1e-7 if not given or given as zero
%   Options(2) Maximal iterations
%            default 2000 if not given or given as zero
%   Options(3) Initialization method
%            A rather slow initialization method is used per default
%            but it pays to investigate in avoiding local minima.
%            Experience may point to faster methods (set Options(3)
%            to 1 or 2). You can also change the number of refits etc.
%            in the beginning of the m-file
%            0 => best of 10 runs of maximally 80 iterations (default)
%            1 => based on SVD
%            2 => random numbers
%   Options(4) Cross-validation
%            0 => no cross-validation
%            1 => cross-validation splitting in 7 segments
%   Options(5) show output
%            0 => show standard output on screen
%            1 => hide all output to screen

Constraints = [2 0]; %This constraint [2 0] produces unique results.
Options = [1e-7, 2000, 1, 0, 1]; %based on SVD speed up the process and give slightly better results.

[A,H,C,P] = parafac2(X, r, Constraints, Options);

end