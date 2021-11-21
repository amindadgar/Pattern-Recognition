function [X,Y] = random_two_variate_uniform(a, b, size)
%random_multivariate_uniform Calculate a multivariable uniform distibution
%a is the lower bound
%b is the higher bound in uniform
%size is the size of the uniform variables
%X is a two multivariable uniform distribution
%Y is two multivariable uniform distribution

x1 = unifrnd(a, b, size);
x2 = unifrnd(a,b, size);
X = meshgrid(x1, x2);

% Again calculate new uniform distribution for Y
x1 = unifrnd(a, b, size);
x2 = unifrnd(a,b, size);
Y = meshgrid(x1, x2);


end