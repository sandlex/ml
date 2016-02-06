function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X=12x1
%y=12x1
%theta=9x1
pred = X * theta;
err = (pred - y) .^ 2;
tterm =  theta(2:size(theta));
J = 1 / (2 * m) * sum(err) + (lambda / (2 * m)) * (tterm' * tterm);

grad = (1 / m) * (X' * (pred - y)) + (lambda / m) .* theta;
grad(1) = grad(1) - (lambda / m) .* theta(1);
% =========================================================================

grad = grad(:);

end
