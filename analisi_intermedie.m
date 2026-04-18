clear
clc
close all

load('preprocessed_data.mat');

%% scatter carico-ora del giorno

scatter(data.TIMESTAMP, data.LOAD);
%% fitting modello con tutte le 25 temperature

temp_matrix_trainval = temp_matrix(1:n+n_v, :);
phi_25_train = [ones(n + n_v, 1), temp_matrix_trainval];

[thetaLS_25, std_error_25, var_cap_25, var_thetaLS_25] = lscov(phi_25_train, data_trainval.LOAD);

% valori altalenanti dei parametri stimati
figure(12)
bar(thetaLS_25(2:end)); 
grid on;
title('Valori dei coefficienti \theta_1...\theta_{25}');

%% prestazioni sul test set

temp_matrix_test = table2array(data_test(:, 3:27));
load_test = data_test.LOAD;

phi_25_test = [ones(height(data_test), 1), temp_matrix_test];

% Predizione del carico
load_cap_25_test = phi_25_test * thetaLS_25;

epsilon_25_test = load_test - load_cap_25_test;


RMSE_25_test = sqrt(mean(epsilon_25_test.^2));
MAPE_25_test = mean(abs(epsilon_25_test ./ load_test)) * 100;

cond_25 = cond(phi_25_train);

fprintf('\n--- RISULTATI MODELLO 25 TEMPERATURE ---\n');
fprintf('RMSE sul Test Set: %.4f MW\n', RMSE_25_test);
fprintf('MAPE sul Test Set: %.2f%%\n', MAPE_25_test);
fprintf('Condition Number della matrice: %.2e\n', cond_25);

%% Modello LOAD vs TIMESTAMP 

t_train = data_trainval.TIMESTAMP;
load_train = data_trainval.LOAD;

phi_ora = [ones(n + n_v, 1), t_train, t_train.^2, t_train.^3];

% Stima dei coefficienti
[theta_ora, std_err_ora] = lscov(phi_ora, load_train);

% Predizione sul set di test per vedere quanto perde senza temperatura
t_test = data_test.TIMESTAMP;
phi_ora_test = [ones(length(t_test), 1), t_test, t_test.^2, t_test.^3];
load_pred_ora = phi_ora_test * theta_ora;

% Metriche
rmse_ora = sqrt(mean((data_test.LOAD - load_pred_ora).^2));
mape_ora = mean(abs((data_test.LOAD - load_pred_ora)./data_test.LOAD)) * 100;

fprintf('--- Modello Solo Ora ---\n');
fprintf('RMSE: %.2f MW\n', rmse_ora);
fprintf('MAPE: %.2f%%\n', mape_ora);


