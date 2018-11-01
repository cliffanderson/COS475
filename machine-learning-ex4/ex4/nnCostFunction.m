function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% FROM LAB 3
% My Notes:
% 
% X -> (m x 400) matrix of input training set
% Theta1 -> (25 x 401) matrix of weights from input to hidden layer
% Theta2 -> (10 x 26) matrix of weights from hidden layer to output layer
%
% z -> vector of weights times activations of the previous layer
% ex) z2 = Theta1 * a1
%

% Add ones to the X data matrix, for bias
X = [ones(m, 1) X];

costSum = 0;
    
for i = 1:m
    
    % Init the actual
    actual = zeros(num_labels, 1);
    actual(y(i)) = 1;

    
    % ----- FEED FORWARD ----- %
    
    % LAYER 2 - Feed forward
    % 25 Nodes
    
    % Calculating z2 -> (25 x 1) vector
    z2 = Theta1 * X(i,:)';
    % Pass z2 into activation function
    a2 = sigmoid(z2);  
    % Add a one to a2, for bias
    a2 = [1 ; a2];      % Now a (26 x 1) vector

    
    % OUTPUT LAYER - Feed forward
    % 10 Output Nodes

    % Calculating z3 -> (10 x 1) vector
    z3 = Theta2 * a2;
    % Pass z3 into activation function
    a3 = sigmoid(z3);       % Probabilities vector
    
    
    % Calculate cost for the network
    costSum = costSum + (actual' .* -1 * log(a3) - (1 - actual') * log(1 - a3));
   
    
    
    
    % ----- ERROR CALCULATION ----- %
   
    % CALCULATE ERROR - OUTPUT LAYER    
    error3 = a3 - actual;
   
    
    % CALCULATE ERROR - HIDDEN LAYER
    
    % make z2 26x1 to conform to matrix sizes
    z2 = [0; z2];
   
    error2 = (Theta2' * error3) .* sigmoidGradient(z2);
    error2(1) = [];
    
    
    
    
    
    
    % ----- BACKPROPAGATION ----- %
    
    % BACKPROPAGATION - Theta2
    Theta2_grad = Theta2_grad + error3 * a2';
    
    % BACKPROPAGATION - Theta1
    a1 = X(i,:)';   % Get the values for the input layer (we don't 
                    % explicitly save them during feed forward
    Theta1_grad = Theta1_grad + error2 * a1';
    
end

% Finalize cost
costSum = costSum / m;
J = costSum;


% Regularize the cost if regularization is enabled
if lambda ~= 0
       
   % Copy Theta1 and Theta2 in order to remove the first column
   new_Theta1 = Theta1;
   new_Theta2 = Theta2;
   
   new_Theta1(:,1) = [];
   new_Theta2(:,1) = [];
   
   part1 = sum(sum(new_Theta1 .^ 2));
   part2 = sum(sum(new_Theta2 .^ 2));
   
   regularizedCost = (lambda / (2*m)) * (part1 + part2);
   
   % Add it to the cost
   J = J + regularizedCost;

end

% Finalize gradients
Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;


% Use regularization on the gradients if it is enabled
if lambda ~= 0
    
    regTerm1 = (lambda/m).*new_Theta1;
    regTerm1 = [zeros(hidden_layer_size,1) regTerm1];
   
    regTerm2 = (lambda/m).*new_Theta2;
    regTerm2 = [zeros(num_labels,1) regTerm2];

    Theta1_grad = Theta1_grad + regTerm1;
    Theta2_grad = Theta2_grad + regTerm2;
    
    % TODO: don't regularize bias
    %Theta2_grad = Theta2_grad + Theta2 .* (lambda/m);
    %Theta1_grad = Theta1_grad + Theta1 .* (lambda/m);
    
    % Replace regularized bias weight with original
   
    
end

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
