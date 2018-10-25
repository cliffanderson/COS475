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


% ===== FEED FORWARD AND CALCULATE COST ===== %

% Init p for the predictions
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix, for bias
X = [ones(m, 1) X];

costSum = 0;
    
for i = 1:m
    
    real_y = zeros(10, 1);
    real_y(y(i)) = 1;

    
% LAYER 2
% 25 Nodes
    
    % Calculating z2 -> (25 x 1) vector
    z2 = Theta1 * X(i,:)';
    % Pass z2 into activation function
    a2 = sigmoid(z2);  
    % Add a one to a2, for bias
    a2 = [1 ; a2];      % Now a (26 x 1) vector

% OUTPUT LAYER
% 10 Output Nodes

    % Calculating z3 -> (10 x 1) vector
    z3 = Theta2 * a2;
    % Pass z3 into activation function
    a3 = sigmoid(z3);       % Probabilities vector
    
    otherA3 = log(a3);
    yetAnotherA3 = log(1 - a3);
    
    % result = (real_y' .* -1 * otherA3 - (1 - real_y') * yetAnotherA3);
    
    
    costSum = costSum + (real_y' .* -1 * otherA3 - (1 - real_y') * yetAnotherA3);
    % save this value in a vector to sum up later
    
    % Find the max probability in the output vector
    % hi -> the highest probability
    % index -> the index of the highest probability (our label)
    [hi, index] = max(a3);
    
    % Add our predicted label to vector p
    p(i) = index;
    
end

costSum = costSum / m;

J = costSum;


if lambda ~= 0
    
   fprintf('\nComputing regularized cost because lambda is non-zero\n');
   
   % Copy Theta1 and Theta2 in order to remove the first column
   new_Theta1 = Theta1;
   new_Theta2 = Theta2;
   
   new_Theta1(:,1) = [];
   new_Theta2(:,1) = [];
   
   part1 = sum(sum(new_Theta1 .^ 2));
   part2 = sum(sum(new_Theta2 .^ 2));
   
   regularizedCost = (lambda / (2*m)) * (part1 + part2);
   
   fprintf('RegularizedCost: %f\n', regularizedCost);
   
   J = J + regularizedCost;

end





%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
