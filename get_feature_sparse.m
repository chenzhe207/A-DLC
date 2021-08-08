function [ new_feature, Alpha ] = get_feature_sparse(X, Y, train_label, sigma)

c = length(unique(train_label));
rho = 1.1;
max_mu = 1e8;
mu = 1e-1;
tr_num = size(X, 2);
te_num = size(Y, 2);
J = zeros(tr_num, te_num);
Lambda = zeros(tr_num, te_num);
I = eye(tr_num, tr_num);
for i = 1 : 50
    % update Alpha
    Alpha = (X' * X + mu * I) \ (X' * Y - Lambda + mu * J);
    %update J
    J = shrinkage(Alpha + Lambda / mu, sigma / mu);    
    %update Lambda and mu
    Lambda = Lambda + mu * (Alpha - J);
%     mu = min(max_mu, mu * rho);
end

for j = 1 : c
     Mmu = zeros(tr_num, te_num);
     ind = (j == train_label)';
     Mmu(ind, :) = Alpha(ind, :);
     E = Y - X * Mmu;
     r(j, :) = sqrt(sum(E .* E));
     clear ind E
end
% softmax
r = exp(-r);
r = r ./ sum(r);
new_feature = r;
end

