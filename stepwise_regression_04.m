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

X_test_step = [ones(n_t, 1), X_test_candidati(:, in)];

% stepwise + Fourier

omega = 2 * pi / 24;

phi_step_arm = [X_train_step, ...
    sin(omega * data_trainval.TIMESTAMP), cos(omega * data_trainval.TIMESTAMP), ... % 1a Armonica
    sin(2*omega * data_trainval.TIMESTAMP), cos(2*omega * data_trainval.TIMESTAMP)]; % 2a Armonica 

[thetaLS_step_arm, std_error_step_arm] = lscov(phi_step_arm, data_trainval.LOAD);

phi_T_step_arm = [X_test_step, ...
    sin(omega * data_test.TIMESTAMP), cos(omega * data_test.TIMESTAMP), ...
    sin(2*omega * data_test.TIMESTAMP), cos(2*omega * data_test.TIMESTAMP)];

% Predizione
load_cap_T_step_arm = phi_T_step_arm * thetaLS_step_arm;

epsilon_T_step_arm = data_test.LOAD - load_cap_T_step_arm;
RMSE_step_arm_test = sqrt(mean(epsilon_T_step_arm.^2));
mape_step_arm = mean(abs(epsilon_T_step_arm ./ data_test.LOAD)) * 100;
SSR_res_stepf = sum(epsilon_T_step_arm.^2);
SSR_tot_stepf = sum((data_test.LOAD - (mean(data_test.LOAD))).^2);
R2_stepf = 1 - (SSR_res_stepf / SSR_tot_stepf)

fprintf('RMSE stepwise + Armoniche: %.2f\n', RMSE_step_arm_test);
fprintf('MAPE stepwise + Armoniche: %.2f%%\n', mape_step_arm);
fprintf('R^2 stepwise + Armoniche: %.4f\n', R2_stepf);

%% Surfacing stepwise + Fourier
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
scatter(load_cap_T_step_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% bisettrice
limiti = [min([load_cap_T_step_arm, data_test.LOAD]), max([load_cap_T_step_arm, data_test.LOAD])];
plot(limiti, limiti, 'r', 'LineWidth', 2);

grid on;
xlabel('Carico Predetto (MW)');
ylabel('Carico Reale (MW)');
title('Goodness of Fit - stepwise + Fourier');
legend('Previsioni', 'Bisettrice', 'Location', 'NorthWest');

%% subplot polinomio 5° + Fourier vs stepwise + Fourier

figure(29);
set(gcf, 'Position', [100, 100, 1200, 500]); 

% Subplot 1: Polinomio 5° grado + Fourier 
ax1 = subplot(1, 2, 1);
mesh(X_surf, Y_surf, Z_surf)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°C)'); ylabel('Ora del giorno'); zlabel('Carico (MW)');
title(sprintf('Polinomio 5° grado + Fourier\nRMSE: %.2f - MAPE: %.2f%% - R^2: %.4f', RMSE_arm_test, mape_arm, R2_polif));
grid on; view(3);

% Subplot 2: Stepwise + Fourier 
ax2 = subplot(1, 2, 2);
mesh(X_surf_step, Y_surf_step, Z_surf_step)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°C)'); ylabel('Ora del giorno'); zlabel('Carico (MW)');
title(sprintf('Stepwise + Fourier\nRMSE: %.2f - MAPE: %.2f%% - R^2: %.4f', RMSE_step_arm_test, mape_step_arm, R2_stepf));
grid on; view(3);

z_min = min([Z_surf(:); Z_surf_step(:)]);
z_max = max([Z_surf(:); Z_surf_step(:)]);
set([ax1, ax2], 'ZLim', [z_min z_max], 'View', [-35, 30]);

h = linkprop([ax1, ax2], {'View', 'XLim', 'YLim', 'ZLim'});
setappdata(gcf, 'StoreTheLink', h);
sgtitle('Confronto superfici'); 

figure(30); % Nuova figura per GoF
set(gcf, 'Position', [150, 150, 1200, 500]);

% Subplot 1: GoF Stepwise Puro
subplot(1, 2, 1);
scatter(load_cap_T_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3); 
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Polinomio 5° grado +  Fourier');

% Subplot 2: GoF Stepwise + Fourier 
subplot(1, 2, 2);
scatter(load_cap_T_step_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Stepwise + Fourier');

sgtitle('Confronto GOF');