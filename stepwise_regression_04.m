clear
clc
close all

load("preprocessed_data.mat")

%% Stepwise Regression 

X_candidati = [w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2, ...
    w_avg_train.* data_train.TIMESTAMP, w_avg_train.^3, data_train.TIMESTAMP.^3, ...
    (w_avg_train.^2).*data_train.TIMESTAMP, w_avg_train.*(data_train.TIMESTAMP.^2), ...
    w_avg_train.^4, data_train.TIMESTAMP.^4, (w_avg_train.^2).*(data_train.TIMESTAMP.^2)];%, w_avg_train.^5];

% Stepwise Selection
[b, se, pval, in, stats] = stepwisefit(X_candidati, data_train.LOAD, 'display','off');

X_train_step = [ones(n, 1), X_candidati(:, in)]; 
theta_step = lscov(X_train_step, data_train.LOAD);
y_hat_train = X_train_step * theta_step;

% Test sul set di validazione

X_test_candidati = [ w_avg_test, data_test.TIMESTAMP,  w_avg_test.^2, data_test.TIMESTAMP.^2, ...
    w_avg_test.*data_test.TIMESTAMP,  w_avg_test.^3, data_test.TIMESTAMP.^3, ...
    ( w_avg_test.^2).* data_test.TIMESTAMP,  w_avg_test.*(data_test.TIMESTAMP.^2), ...
    w_avg_test.^4, data_test.TIMESTAMP.^4, (w_avg_test.^2).*(data_test.TIMESTAMP.^2)];%, w_avg_test.^5];

X_test_step = [ones(length(w_avg_test), 1), X_test_candidati(:, in)];

y_hat_step = X_test_step * theta_step;
RMSE_step_train = sqrt(mean((data_train.LOAD - y_hat_train).^2))
RMSE_step_test = sqrt(mean((data_test.LOAD - y_hat_step).^2))

%% surfacing stepwise

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23, 24);
[X, Y] = meshgrid(x, y);

x_vec = X(:);
y_vec = Y(:);

phi_grid_candidati = [x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.*y_vec, ...
                      x_vec.^3, y_vec.^3, (x_vec.^2).*y_vec, x_vec.*(y_vec.^2), ...
                      x_vec.^4, y_vec.^4, (x_vec.^2).*(y_vec.^2)];

phi_grid_final = [ones(size(x_vec, 1), 1), phi_grid_candidati(:, in)];

z_cap = phi_grid_final * theta_step;
Z = reshape(z_cap, size(X));

figure(20)
mesh(X, Y, Z)
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);

xlabel('Temperatura Media (°C)')
ylabel('Ora del Giorno')
zlabel('Carico Elettrico (MW)')
title('Superficie con Stepwise Regression')
grid on
