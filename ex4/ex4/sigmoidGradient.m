function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

z_conv = [z(:)];
for i = 1 : size(z_conv, 1),
  z_conv(i) = sigmoid(z_conv(i)) * (1 - sigmoid(z_conv(i)));
end;
g = reshape(z_conv(:), size(z, 1), size(z, 2));

% =============================================================




end
