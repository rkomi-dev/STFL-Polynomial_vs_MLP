clear
clc
close all

load('preprocessed_data.mat');

%% trasformazione timestamp in periodico e normalizzazione

% Trasformazione Periodica
T = 24;
time_sin_trainval = sin(2 * pi * data_trainval.TIMESTAMP / T);
time_cos_trainval = cos(2 * pi * data_trainval.TIMESTAMP / T);

X_trainval_mlp = [w_avg_trainval, time_sin_trainval, time_cos_trainval]; 

% Normalizzazione

[X_trainval_n, settings_x] = mapminmax(X_trainval_mlp'); 
[y_trainval_n, settings_y] = mapminmax(data_trainval.LOAD');

%% Scelta neuroni tramite K-Fold Standard e regola 1-SD
k = 5; 
N_tot = length(y_trainval_n);

% Creazione dei fold 
cv = cvpartition(N_tot, 'KFold', k); 

neuron_range = 1:15; 
rmse_kfold_train_medio = zeros(length(neuron_range), 1);
rmse_kfold_val_medio = zeros(length(neuron_range), 1);
rmse_kfold_val_std = zeros(length(neuron_range), 1);

fprintf('\nInizio K-Fold Standard\n');

for i = 1:length(neuron_range)
    n_hid = neuron_range(i);
    rmse_train_fold = zeros(k, 1);
    rmse_val_fold = zeros(k, 1);
    
    for f = 1:k
        % Indici per il fold corrente
        idx_train = training(cv, f);
        idx_val = test(cv, f);
        
        % Split dei dati
        X_train_fold = X_trainval_n(:, idx_train);
        y_train_fold = y_trainval_n(idx_train);
        X_val_fold = X_trainval_n(:, idx_val);
        y_val_fold = y_trainval_n(idx_val);
        
        % Training della rete
        net_fold = fitnet(n_hid, 'trainlm');
        net_fold.trainParam.showWindow = false;
        net_fold = train(net_fold, X_train_fold, y_train_fold);
        
        % Valutazione Training 
        y_tr_pred = mapminmax('reverse', net_fold(X_train_fold), settings_y);
        y_tr_real = mapminmax('reverse', y_train_fold, settings_y);
        rmse_train_fold(f) = sqrt(mean((y_tr_real - y_tr_pred).^2));
        
        % Valutazione Validazione 
        y_val_pred = mapminmax('reverse', net_fold(X_val_fold), settings_y);
        y_val_real = mapminmax('reverse', y_val_fold, settings_y);
        rmse_val_fold(f) = sqrt(mean((y_val_real - y_val_pred).^2));
    end
    
    % Medie e Deviazione Standard per il numero di neuroni corrente
    rmse_kfold_train_medio(i) = mean(rmse_train_fold);
    rmse_kfold_val_medio(i) = mean(rmse_val_fold);
    rmse_kfold_val_std(i) = std(rmse_val_fold);
    
    fprintf('Neuroni: %d | RMSE Val Medio: %.2f MW (±%.2f)\n', ...
        n_hid, rmse_kfold_val_medio(i), rmse_kfold_val_std(i));
end

% Regola 1-SD 
[min_rmse, idx_min] = min(rmse_kfold_val_medio);
sd_del_min = rmse_kfold_val_std(idx_min);
soglia_tolleranza = min_rmse + sd_del_min;

% modello più semplice che rientra nella soglia
idx_scelto = find(rmse_kfold_val_medio <= soglia_tolleranza, 1, 'first');
n_hid_ottimo = neuron_range(idx_scelto);

fprintf('\nMinimo RMSE: %.2f MW (a %d neuroni)\n', min_rmse, neuron_range(idx_min));
fprintf('Numero neuroni SCELTO (Regola 1-SD): %d\n', n_hid_ottimo);


%% confronto complessità vs RMSE

figure(21)
plot(neuron_range, rmse_kfold_train_medio, '-s', 'LineWidth', 1.5);
hold on
plot(neuron_range, rmse_kfold_val_medio, '-s', 'LineWidth', 1.5);
hold on
yline(soglia_tolleranza, '--r', 'LineWidth', 2);
plot(n_hid_ottimo, rmse_kfold_val_medio(idx_scelto), 'gs', 'MarkerSize', 12, 'LineWidth', 2);
grid on
xlabel('Numero di Neuroni (Hidden Layer)')
ylabel('RMSE (MW)')
legend('Train RMSE', 'Validazione RMSE', 'Soglia di tolleranza')
title('complessità vs RMSE: MLP')

%% prestazioni sul test set

net_finale = fitnet(8, 'trainlm');
net_finale.trainParam.showWindow = false; 

net_finale = train(net_finale, X_trainval_n, y_trainval_n);

time_sin_test = sin(2 * pi * data_test.TIMESTAMP / T);
time_cos_test = cos(2 * pi * data_test.TIMESTAMP / T);
X_test = [w_avg_test, time_sin_test, time_cos_test];

X_test_n = mapminmax('apply', X_test', settings_x);

y_test_pred_n = net_finale(X_test_n);
y_test_pred_mw = mapminmax('reverse', y_test_pred_n, settings_y);
y_test_real_mw = data_test.LOAD'; 

errore = y_test_real_mw - y_test_pred_mw;
MAPE_mlp = mean(abs(errore ./ y_test_real_mw)) * 100;
RMSE = sqrt(mean(errore.^2));

fprintf('--- PERFORMANCE FINALI SUL TEST SET ---\n');
fprintf('RMSE: %.4f MW\nMAPE: %.2f%%\n', RMSE, MAPE_mlp);

%% boxplot dei residui
figure;
boxplot(errore, data_test.TIMESTAMP);
title('Residui sul Test Set (Addestramento su 85%)');
grid on;
%% surfacing con MLP a 8 neuroni

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23.9, 100); 
[X, Y] = meshgrid(x, y);

T = 24;
Y_sin = sin(2 * pi * Y(:) / T);
Y_cos = cos(2 * pi * Y(:) / T);

phi_vec = [X(:), Y_sin, Y_cos]'; 

phi_vec_n = mapminmax('apply', phi_vec, settings_x);

z_cap_n = net_finale(phi_vec_n);
z_cap = mapminmax('reverse', z_cap_n, settings_y); 

Z = reshape(z_cap, size(X));

figure(22)
mesh(X, Y, Z)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
xlabel('Temperatura Media (°C)')
ylabel('Ora del Giorno')
zlabel('Carico Elettrico (MW)')
title('Superficie MLP con 8 neuroni');
grid on

%% test di continuità della superficie (tiene conto delle circolarità del tempo)

% temperatura di test
temp_test = 20; 

% 23:59 (Quasi mezzanotte)
t1 = 23.99;
input1 = [temp_test, sin(2*pi*t1/24), cos(2*pi*t1/24)]';
input1_n = mapminmax('apply', input1, settings_x);
y1 = mapminmax('reverse', net_finale(input1_n)', settings_y);

% 00:01 (Appena dopo mezzanotte)
t2 = 0.01;
input2 = [temp_test, sin(2*pi*t2/24), cos(2*pi*t2/24)]';
input2_n = mapminmax('apply', input2, settings_x);
y2 = mapminmax('reverse',  net_finale(input2_n)', settings_y);

fprintf('Carico alle 23:59: %.4f MW\n', y1);
fprintf('Carico alle 00:01: %.4f MW\n', y2);
fprintf('Salto di continuità: %.4f MW\n', abs(y1-y2));

%% goodness of fit

figure(23)
scatter(y_test_pred_mw, y_test_real_mw, 20, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% bisettrice
limiti = [min([y_test_pred_mw, y_test_real_mw]), max([y_test_pred_mw, y_test_real_mw])];
plot(limiti, limiti, 'r', 'LineWidth', 2);

grid on;
xlabel('Carico Predetto (MW)');
ylabel('Carico Reale (MW)');
title('Goodness of Fit - MLP');
legend('Previsioni MLP', 'Bisettrice', 'Location', 'NorthWest');

%% confronto finale stepwise + Fourier vs MLP

%% subplot stepwise vs MLP

figure('Name', 'Confronto Superfici di Risposta', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.5]);

% Subplot 1: stepwise + Armoniche 
ax1 = subplot(1, 2, 1);
mesh(X_surf_step, Y_surf_step, Z_surf_step)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°C)')
ylabel('Ora del giorno')
zlabel('Carico elettrico (MW)')
title(sprintf('Stepwise + Fourier\nRMSE: %.2f - MAPE: %.2f%%', RMSE_step_arm_test, mape_step_arm));
grid on

% Subplot 2: MLP a 8 Neuroni
ax2 = subplot(1, 2, 2);
mesh(X, Y, Z) 
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°C)')
ylabel('Ora del giorno')
zlabel('Carico elettrico (MW)')
title(sprintf('MLP con 8 Neuroni\n RMSE: %.2f - MAPE: %.2f%%', RMSE, MAPE));
grid on

z_min = min([Z_surf_step(:); Z(:)]);
z_max = max([Z_surf_step(:); Z(:)]);
set([ax1, ax2], 'ZLim', [z_min z_max], 'View', [-35, 30]);

h = linkprop([ax1, ax2], {'View', 'XLim', 'YLim', 'ZLim'});
setappdata(gcf, 'StoreTheLink', h); 

sgtitle('Confronto superfici: Stepwise + Fourier vs MLP');

%% subplot gof stepwise vs mlp

figure; 
set(gcf, 'Position', [150, 150, 1200, 500]);

all_values = [y_hat_step(:); data_test.LOAD(:); y_test_pred_mw(:); y_test_real_mw(:)];
min_val = min(all_values);
max_val = max(all_values);
limiti = [min_val, max_val];

% Subplot 1: GoF Stepwise Puro 
subplot(1, 2, 1);
scatter(y_hat_step, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3); 
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Stepwise + Fourier');

% Subplot 2: GoF Stepwise + Fourier 
subplot(1, 2, 2);
scatter(y_test_pred_mw, y_test_real_mw, 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: MLP con 8 neuroni');

sgtitle('Confronto GOF');