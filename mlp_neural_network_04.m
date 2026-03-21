clear
clc
close all

load('preprocessed_data.mat');

% Preparazione Input
X_train_mlp = [w_avg_train, data_train.TIMESTAMP];
X_test_mlp = [w_avg_test, data_test.TIMESTAMP];

% Normalizzazione
[X_train_n, settings] = mapminmax(X_train_mlp');
X_test_n = mapminmax('apply', X_test_mlp', settings);
y_train_n = data_train.LOAD';

% Training MLP (14 neuroni)
net = fitnet(14, 'trainlm');
net.trainParam.showWindow = false;
[net, tr] = train(net, X_train_n, y_train_n);

% Previsione
y_hat_mlp = net(X_test_n)';
RMSE_mlp = sqrt(mean((data_test.LOAD - y_hat_mlp).^2));

fprintf('RMSE MLP: %.2f MW\n', RMSE_mlp);

% Visualizzazione Superficie MLP
x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23, 24);
[X, Y] = meshgrid(x, y);
input_grid = [X(:), Y(:)];
input_grid_n = mapminmax('apply', input_grid', settings);
Z_mlp = reshape(net(input_grid_n), size(X));

figure; mesh(X, Y, Z_mlp); title('Superficie con MLP a 14 neuroni');
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
save('mlp_model.mat', 'net', 'settings', 'y_hat_mlp');