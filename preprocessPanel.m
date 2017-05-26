function [DATA, nnow, gt] = preprocessPanel(DATA, q, r, p, nnow);

% make a copy to avoid corrupting the original data
x = DATA(:,:);
[T,N] = size(x);
nnowcast = floor(T*nnow); % In most cases we only nowcast the last 1/4

% Adjust for outliers and missing data.
% Positions of spatial series (distance to work and distance to home)
% For the two-step model, to make fair comparisons with other methods, we
% finally ignored the spatial series (only which may contain NaN values).
dw = N-1; dh = N-2;
clear xc
for j = [dh,dw] % Address the spatial missing values in training/historical part
    %This function first replaces the outliers and missing values
    %with the median, and then runs an order-3 moving average on 
    %the completed vector, and finally replaces the outliers and 
    %missing values with the value in the same position in the 
    %averaged vector.
    xc(:,j) = outliers_correction(x(:,j));
end;

% To simulate the real-time data release or signal arrival.
% adjust only data up to the last week to preserve the unbalanced structure
% In fact, at the end of the sample the missing data are due to the timing of data
% releases and not to coding, download, computing errors.
x(1:end-nnowcast,[dh,dw]) = xc(1:end-nnowcast,[dh,dw]); % 02/06/2016 to experiment with later
%x(1:end,[dh,dw]) = xc(1:end,[dh,dw]); %Probably needed when we have to complete all the missing locations including those in nowcasted part

% Check whether the balanced panel is full rank or not.
% If not, perform feature selection.
bpx_tmp = balancePanel(x(:, 1:end-1),q,r,p,nnow);
act_label = x(:,end);
x_tmp = x(1:end, var(bpx_tmp)>0);
x = [x_tmp act_label];

[T,N] = size(x);
gt = x(end-nnowcast+1:end, N);
x(end-nnowcast+1:end, N) = NaN*ones(nnowcast,1);
if( N-1 <= q+r )
    warning(['Bad panel num. of valid series ', num2str(N-1), ' <= num. of factors and shocks ', num2str(q+r)]);
end

DATA = x;
%--------------------- End Preprocessing Steps --------------------------