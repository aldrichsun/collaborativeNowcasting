function [Z,Jout,Jmis] = outliers_correction(X);
%
% This function adjusts for outliers and missing observations. 
% It first replaces the outliers and missing values with the 
% median, and then run an order-3 moving average on the completed vector, 
% and finally replace the outliers and missing values with the 
% corresponding value in the averaged vector.
%
% We could basically ignore outliers, and missing values for nowcasting are left there.
%
% Code based on: "Nowcasting: The Real Time Informational Content of Macroeconomic Data",
% Domenico Giannone, Lucrezia Reichlin & David Small, Journal of Monetary Economics.

Jmis = isnan(X);
x = X(~Jmis);
a = sort(x); % For vectors, sort(X) sorts the elements of X in ascending order. 
             % For matrices, sort(X) sorts each column of X in ascending order. 
             % Here, we sort the values in each series in order to find the median 
             % and use the median to replace the missing values.
T = size(x,1);

% define outliers as those obs. that exceed 4 times the interquartile distance
%Jout = (abs(X-a(round(T/2))) > 4*abs(a(round(T*1/4))-a(round(T*3/4))));

Z = X;
Z(Jmis) = a(round(T/2)); % put the median in place of missing values
%Z(Jout) = a(round(T/2)); % put the median in place of outliers

Zma = MAcentered(Z,3); % MAcentered (at bottom of this file) computes the moving average of order k (3 here).

%Z(Jout) = Zma(Jout);
%Z(Jmis) = Zma(Jmis);
Z = Zma; % 01/08/2015: to compare uniformfeatures


% This function computes moving average (MA) of order k
function x_ma = MAcentered(x,k_ma);
xpad = [x(1,:).*ones(k_ma,1); x; x(end,:).*ones(k_ma,1)];
for j = k_ma+1:length(xpad)-k_ma;
    x_ma(j-k_ma,:) = mean(xpad(j-k_ma:j+k_ma,:));
end;