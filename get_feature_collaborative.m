function [ new_feature, Beta ] = get_feature_collaborative(X, Y, train_label, cons)

c = length(unique(train_label));
tr_num = size(X, 2);
te_num = size(Y, 2);
Beta = cons * Y;

for j = 1 : c
     Mmu = zeros(tr_num, te_num);
     ind = (j == train_label)';
     Mmu(ind, :) = Beta(ind, :);
     E = Y - X * Mmu;
     r(j, :) = sqrt(sum(E .* E));
     clear ind E
end
% softmax
r = exp(-r);
r = r ./ sum(r);
new_feature = r;

end
