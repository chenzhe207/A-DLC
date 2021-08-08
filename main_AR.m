clear all
clc
close all

addpath('D:\0 学习 ☆☆☆\数据库');

load randomprojection_AR
DATA = DATA./ repmat(sqrt(sum(DATA .* DATA)), [size(DATA, 1) 1]); %normalize
c = length(unique(Label));

train_num = 10;
layer_num = 10;

for ii = 1 : 10
train_data = []; test_data = []; 
train_label = []; test_label = [];
for i = 1 : c
    index = find(Label == i); 
    randindex = index(randperm(length(index)));
    train_data = [train_data DATA(:,randindex(1 : train_num))];
    train_label = [train_label  Label(randindex(1 : train_num))];
    test_data = [test_data DATA(:, randindex(train_num + 1 : end))];
    test_label = [test_label  Label(randindex(train_num + 1 : end))];
    train_index(i, :) = find(train_label == i);
end
for i = 1 : size(train_data, 2)
    a = train_label(i);
    Htr(a, i) = 1;
end 

lambda = 1e-1;
sigma = 1e-5;
tau = 1e-4;
    
%% preprocessing
Q = inv(train_data * train_data' + lambda * eye(size(train_data, 1))) * train_data * Htr';
train_new = Q' * train_data; % new training feature c*n
test_new = Q' * test_data; % new test feature c*1
train_new = normcols(train_new); % normalize
test_new = normcols(test_new); % normalize

%% sparse-collaborative alternative cascaded representation (SCACR)
[Layer_acc(ii, :), Result_index, Test_colla_coeff, Test_colla_soft, Test_sparse_coeff, Test_sparse_soft] = A_DCM(train_new, test_new, train_label, test_label, layer_num, sigma, tau);
end 
% save DATA_AR_N8_K10 Layer_acc

% end
t = Test_colla_soft{10};
tt = t(:,689);
plot(tt(1:50))

