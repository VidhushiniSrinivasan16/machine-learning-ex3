function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];
propagation1=sigmoid(X*Theta1');
disp(size(propagation1,1));
disp(size(propagation1,2));
propagation1=[ones(size(propagation1,1),1) propagation1];
propagation2=sigmoid(propagation1*Theta2');
disp(size(propagation2,1));
disp(size(propagation2,2));
max_index=0;
for i=1:m 
[val,index]=max(propagation2(i,:));
if(index~=0)
p(i,1)=index;
else
p(i,1)=num_labels;
end





% =========================================================================


end
