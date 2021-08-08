function [ Layer_acc, Result_index, Test_colla_coeff, Test_colla_soft, Test_sparse_coeff, Test_sparse_soft  ] = A_DCM(X, Y, train_label, test_label, layer_num, sigma, tau)

Layer_acc = zeros(1, layer_num);

for i = 1 : layer_num
    % collaborative
    cons = inv(X' * X + tau * eye(size(X, 2))) * X';
    [train_colla_soft, train_colla_coeff] = get_feature_collaborative(X, X, train_label, cons);
    [test_colla_soft, test_colla_coeff] = get_feature_collaborative(X, Y, train_label, cons);
    Test_colla_coeff{i} = test_colla_coeff;
    Test_colla_soft{i} = test_colla_soft;
    X = [X; train_colla_soft];
    Y = [Y; test_colla_soft];   
    X = normcols(X); % normalize
    Y = normcols(Y); % normalize

    % sparse
    [train_sparse_soft, train_spars_coeff] = get_feature_sparse(X, X, train_label, sigma);
    [test_sparse_soft, test_spars_coeff] = get_feature_sparse(X, Y, train_label, sigma);
    Test_sparse_coeff{i} = test_spars_coeff;
    Test_sparse_soft{i} = test_sparse_soft;
    [~, ind] = max(test_sparse_soft);
    Layer_acc(i) = sum(ind == test_label) / size(test_label, 2)
    Result_index(i, :) = double(ind == test_label);
    
    X = [X; train_sparse_soft];
    Y = [Y; test_sparse_soft]; 
    X = normcols(X); % normalize
    Y = normcols(Y); % normalize

end
end

