% Exercise 3 -- Multivariate Linear Regression

clear all; close all; clc

x = load('ex3x.dat'); 
y = load('ex3y.dat');

m = length(y);

% Add intercept term to x
x = [ones(m, 1), x];

% Save a copy of the unscaled features for later
x_unscaled = x;

% Scale features and set them to zero mean
mu = mean(x);
sigma = std(x);
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

% Prepare for plotting
figure;
% plot each alpha's data points in a different style
% braces indicate a cell, not just a regular array.
plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--'};


% Gradient Descent 
alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];
MAX_ITR = 100;
% this will contain my final values of theta
% after I've found the best learning rate
theta_grad_descent = zeros(size(x(1,:))); 

for i = 1:length(alpha)
    theta = zeros(size(x(1,:)))'; % initialize fitting parameters
    J = zeros(MAX_ITR, 1);
    for num_iterations = 1:MAX_ITR
        % Calculate the J term
        J(num_iterations) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
        
        % The gradient
        grad = (1/m) .* x' * ((x * theta) - y);
        
        % Here is the actual update
        theta = theta - alpha(i) .* grad;
    end
    % Now plot the first 50 J terms
    plot(0:49, J(1:50), char(plotstyle(i)), 'LineWidth', 2)
    hold on
    
    % After some trial and error, I find alpha=1
    % is the best learning rate and converges
    % before the 100th iteration
    %
    % so I save the theta for alpha=1 as the result of 
    % gradient descent
    if (alpha(i) == 1)
        theta_grad_descent = theta;
    end
end
legend('0.01','0.03','0.1', '0.3', '1', '1.3')
xlabel('Number of iterations')
ylabel('Cost J')

% force Matlab to display more than 4 decimal places
% formatting persists for rest of this session
format long

% Display gradient descent's result
theta_grad_descent

% Estimate the price of a 1650 sq-ft, 3 br house
price_grad_desc = dot(theta_grad_descent, [1, (1650 - mu(2))/sigma(2),...
                    (3 - mu(3))/sigma(3)])

% Calculate the parameters from the normal equation
theta_normal = (x_unscaled' * x_unscaled)\x_unscaled' * y

%Estimate the house price again
price_normal = dot(theta_normal, [1, 1650, 3])

