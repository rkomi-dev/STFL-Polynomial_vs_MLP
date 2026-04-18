clear
clc
close all

load("preprocessed_data.mat")

%% Stepwise Regression 

X_candidati = [w_avg_trainval, data_trainval.TIMESTAMP, w_avg_trainval.^2, data_trainval.TIMESTAMP.^2, ...
    w_avg_trainval.* data_trainval.TIMESTAMP, w_avg_trainval.^3, data_trainval.TIMESTAMP.^3, ...
    (w_avg_trainval.^2).*data_trainval.TIMESTAMP, w_avg_trainval.*(data_trainval.TIMESTAMP.^2), ...
    w_avg_trainval.^4, data_trainval.TIMESTAMP.^4, (w_avg_trainval.^2).*(data_trainval.TIMESTAMP.^2), ...
    w_avg_trainval.^5, data_trainval.TIMESTAMP.^5, ...
    (w_avg_trainval.^4).*data_trainval.TIMESTAMP, (w_avg_trainval.^3).*(data_trainval.TIMESTAMP.^2), ...
    (w_avg_trainval.^2).*(data_trainval.TIMESTAMP.^3), w_avg_trainval.*(data_trainval.TIMESTAMP.^4)];%, ...
   % w_avg_trainval.^6, data_trainval.TIMESTAMP.^6];

% Stepwise Selection
[b, se, pval, in, stats] = stepwisefit(X_candidati, data_trainval.LOAD, 'penter', 0.01, 'display','off');

X_train_step = [ones((n + n_v), 1), X_candidati(:, in)]; 
theta_step = lscov(X_train_step, data_trainval.LOAD);
y_hat_train = X_train_step * theta_step;

% Test sul set di validazione

X_test_candidati = [w_avg_test, data_test.TIMESTAMP, w_avg_test.^2, data_test.TIMESTAMP.^2, ...
    w_avg_test.*data_test.TIMESTAMP, w_avg_test.^3, data_test.TIMESTAMP.^3, ...
    (w_avg_test.^2).*data_test.TIMESTAMP, w_avg_test.*(data_test.TIMESTAMP.^2), ...
    w_avg_test.^4, data_test.TIMESTAMP.^4, (w_avg_test.^2).*(data_test.TIMESTAMP.^2), ...
    w_avg_test.^5, data_test.TIMESTAMP.^5, ...
    (w_avg_test.^4).*data_test.TIMESTAMP, (w_avg_test.^3).*(data_test.TIMESTAMP.^2), ...
    (w_avg_test.^2).*(data_test.TIMESTAMP.^3), w_avg_test.*(data_test.TIMESTAMP.^4)];%, ...
    %w_avg_test.^6, data_test.TIMESTAMP.^6]; 
    
    %commentato 6° grado perchè scende in train ma sale in validazione

X_val_step = [ones(n_t, 1), X_test_candidati(:, in)];
y_hat_step = X_val_step * theta_step;
RMSE_step_train = sqrt(mean((data_trainval.LOAD - y_hat_train).^2));
RMSE_step_test = sqrt(mean((data_test.LOAD - y_hat_step).^2));
MAPE = mean(abs((data_test.LOAD - y_hat_step) ./ data_test.LOAD)) * 100;
%% surfacing stepwise

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23.9, 100);
[X, Y] = meshgrid(x, y);

x_vec = X(:);
y_vec = Y(:);

phi_grid_candidati = [x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.*y_vec, ...
                      x_vec.^3, y_vec.^3, (x_vec.^2).*y_vec, x_vec.*(y_vec.^2), ...
                      x_vec.^4, y_vec.^4, (x_vec.^2).*(y_vec.^2), x_vec.^5, y_vec.^5, ...
    (x_vec.^4).* y_vec, (x_vec.^3).*( y_vec.^2), ...
    (x_vec.^2).*( y_vec.^3), x_vec.*( y_vec.^4)];

phi_grid_final = [ones(size(x_vec, 1), 1), phi_grid_candidati(:, in)];

z_cap = phi_grid_final * theta_step;
Z = reshape(z_cap, size(X));

figure(26)
mesh(X, Y, Z)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);

xlabel('Temperatura Media (°C)')
ylabel('Ora del Giorno')
zlabel('Carico Elettrico (MW)')
title('Superficie con Stepwise Regression');
grid on

%% stepwise + Fourier

omega = 2 * pi / 24;

phi_step_arm = [X_train_step, ...
    sin(omega * data_trainval.TIMESTAMP), cos(omega * data_trainval.TIMESTAMP), ... % 1a Armonica
    sin(2*omega * data_trainval.TIMESTAMP), cos(2*omega * data_trainval.TIMESTAMP)]; % 2a Armonica 

[thetaLS_step_arm, std_error_step_arm] = lscov(phi_step_arm, data_trainval.LOAD);

phi_V_step_arm = [X_val_step, ...
    sin(omega * data_test.TIMESTAMP), cos(omega * data_test.TIMESTAMP), ...
    sin(2*omega * data_test.TIMESTAMP), cos(2*omega * data_test.TIMESTAMP)];

% Predizione
load_cap_V_step_arm = phi_V_step_arm * thetaLS_step_arm;

epsilon_V_step_arm = data_test.LOAD - load_cap_V_step_arm;
RMSE_step_arm_test = sqrt(mean(epsilon_V_step_arm.^2));
mape_step_arm = mean(abs(epsilon_V_step_arm ./ data_test.LOAD)) * 100;

fprintf('--- Confronto Modelli Polinomiali ---\n');
fprintf('MAPE Polinomio 5° Puro: %.2f%%\n', MAPE);
fprintf('MAPE Polinomio + Armoniche: %.2f%%\n', mape_step_arm);

%% Surfacing Modello Polinomiale + Armoniche 
x_surf_step = linspace(min(w_avg_trainval), max(w_avg_trainval), 50); 
y_surf_step = linspace(0, 23.9, 100); 
[X_surf_step, Y_surf_step] = meshgrid(x_surf_step, y_surf_step);

T_period = 24;
omega_surf = 2 * pi / T_period;
Y_sin1 = sin(omega_surf * Y_surf_step(:));
Y_cos1 = cos(omega_surf * Y_surf_step(:));
Y_sin2 = sin(2 * omega_surf * Y_surf_step(:));
Y_cos2 = cos(2 * omega_surf * Y_surf_step(:));

phi_grid_poly = [ones(numel(X_surf_step), 1), X_surf_step(:), Y_surf_step(:), ...
    X_surf_step(:).^2, Y_surf_step(:).^2, X_surf_step(:).*Y_surf_step(:), ...
    X_surf_step(:).^3, Y_surf_step(:).^3, (X_surf_step(:).^2).*Y_surf_step(:), ...
    X_surf_step(:).*(Y_surf_step(:).^2), X_surf_step(:).^4, Y_surf_step(:).^4, (X_surf_step(:).^2).*(Y_surf_step(:).^2), ...
    X_surf_step(:).^5, Y_surf_step(:).^5, ( X_surf_step(:).^4).* Y_surf_step(:), ( X_surf_step(:).^3).*( Y_surf_step(:).^2), ...
    ( X_surf_step(:).^2).*( Y_surf_step(:).^3),  X_surf_step(:).*( Y_surf_step(:).^4)];

phi_grid_arm = [Y_sin1, Y_cos1, Y_sin2, Y_cos2];

phi_grid_tot = [phi_grid_poly, phi_grid_arm];

z_cap_poly = phi_grid_tot * thetaLS_step_arm;
Z_surf_step = reshape(z_cap_poly, size(X_surf_step));

figure(27) 
mesh(X_surf_step, Y_surf_step, Z_surf_step)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
xlabel('Temperatura Media (°C)')
ylabel('Ora del Giorno')
zlabel('Carico Elettrico (MW)')
title('Superficie stepwise + Fourier');
grid on

%% goodness of fit

figure(28)
scatter(load_cap_V_step_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% bisettrice
limiti = [min([load_cap_V_step_arm, data_test.LOAD]), max([load_cap_V_step_arm, data_test.LOAD])];
plot(limiti, limiti, 'r', 'LineWidth', 2);

grid on;
xlabel('Carico Predetto (MW)');
ylabel('Carico Reale (MW)');
title('Goodness of Fit - stepwise + Fourier');
legend('Previsioni', 'Bisettrice', 'Location', 'NorthWest');

%% subplot stepwise vs stepwise + Fourier

figure(29);
set(gcf, 'Position', [100, 100, 1200, 500]); 

% Subplot 1: Solo Stepwise Polinomiale 
subplot(1, 2, 1);
mesh(X, Y, Z)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°C)'); ylabel('Ora del giorno'); zlabel('Carico (MW)');
title(sprintf('Stepwise Puro\nRMSE: %.2f - MAPE: %.2f%%', RMSE_step_test, MAPE));
grid on; view(3);

% Subplot 2: Stepwise + Fourier 
subplot(1, 2, 2);
mesh(X_surf_step, Y_surf_step, Z_surf_step)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°C)'); ylabel('Ora del giorno'); zlabel('Carico (MW)');
title(sprintf('Stepwise + Fourier\nRMSE: %.2f - MAPE: %.2f%%', RMSE_step_arm_test, mape_step_arm));
grid on; view(3);

sgtitle('Confronto superfici'); 

figure(30); % Nuova figura per GoF
set(gcf, 'Position', [150, 150, 1200, 500]);

% Subplot 1: GoF Stepwise Puro
subplot(1, 2, 1);
scatter(y_hat_step, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3); 
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Stepwise Puro');

% Subplot 2: GoF Stepwise + Fourier 
subplot(1, 2, 2);
scatter(load_cap_V_step_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Stepwise + Fourier');

sgtitle('Confronto GOF');