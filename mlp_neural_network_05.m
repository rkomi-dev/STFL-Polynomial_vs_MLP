clear
clc
close all

load('preprocessed_data.mat');

%% trasformazione timestamp in periodico e normalizzazione

% Trasformazione Periodica
T = 24;
time_sin_train = sin(2 * pi * data_train.TIMESTAMP / T);
time_cos_train = cos(2 * pi * data_train.TIMESTAMP / T);
time_sin_val = sin(2 * pi * data_val.TIMESTAMP / T);
time_cos_val = cos(2 * pi * data_val.TIMESTAMP / T);

X_train_mlp = [w_avg_train, time_sin_train, time_cos_train]; 
X_val_mlp = [w_avg_val, time_sin_val, time_cos_val];

% Normalizzazione

[X_train_n, settings_x] = mapminmax(X_train_mlp'); 
X_val_n = mapminmax('apply', X_val_mlp', settings_x);

[y_train_n, settings_y] = mapminmax(data_train.LOAD');

%% scelta neuroni tramite k-fold sequenziale

k = 5; % Numero di fold
punti_taglio = floor(linspace((n - n_t)/2, (n - n_t)*0.9, k)); 

neuron_range = 1:15; 

rmse_kfold_train_medio = zeros(length(neuron_range), 1);
rmse_kfold_test_medio = zeros(length(neuron_range), 1);

fprintf('\nInizio K-Fold Sequenziale...\n');
for i = 1:length(neuron_range)
    n_hid = neuron_range(i);
    rmse_train_fold = zeros(k, 1);
    rmse_val_fold = zeros(k, 1);
    
    for f = 1:k
        idx_fine_train = punti_taglio(f);
        % Definiamo la finestra di test (es. i successivi 500 punti o fino alla fine)
        idx_fine_val = min(idx_fine_train + floor((n - n_t)/10), n - n_t);
        
        % Split Sequenziale (Il Test è sempre nel futuro rispetto al Train)
        X_train_fold = X_train_n(:, 1:idx_fine_train);
        y_train_fold = y_train_n(1:idx_fine_train);
        X_val_fold = X_train_n(:, idx_fine_train+1 : idx_fine_val);
        y_val_fold = y_train_n(idx_fine_train+1 : idx_fine_val);
        
        % Training rapido sul fold
        net_fold = fitnet(n_hid, 'trainlm');
        net_fold.trainParam.showWindow = false;
        net_fold = train(net_fold, X_train_fold, y_train_fold);
        
        y_train_pred_mw = mapminmax('reverse', net_fold(X_train_fold), settings_y);
        y_train_mw = mapminmax('reverse', y_train_fold, settings_y);
        rmse_train_fold(f) = sqrt(mean((y_train_mw - y_train_pred_mw).^2));

        y_val_pred_mw = mapminmax('reverse', net_fold(X_val_fold), settings_y);
        y_val_mw = mapminmax('reverse', y_val_fold, settings_y);
        rmse_val_fold(f) = sqrt(mean((y_val_mw - y_val_pred_mw).^2));
    end
    
    rmse_kfold_train_medio(i) = mean(rmse_train_fold);
    rmse_kfold_val_medio(i) = mean(rmse_val_fold);
    fprintf('Neuroni: %d | RMSE train Medio K-Fold: %.2f MW |RMSE validazione Medio K-Fold: %.2f MW\n', ...
        n_hid, rmse_kfold_train_medio(i), rmse_kfold_val_medio(i));
end


%% confronto complessità vs RMSE

figure(21)
plot(neuron_range, rmse_kfold_train_medio, '-s', 'LineWidth', 1.5);
hold on
plot(neuron_range, rmse_kfold_val_medio, '-s', 'LineWidth', 1.5);
hold on
grid on
xlabel('Numero di Neuroni (Hidden Layer)')
ylabel('RMSE (MW)')
legend('Train RMSE', 'Validazione RMSE')
title('complessità vs RMSE: MLP')

%% training MLP a 8 neuroni

net = fitnet(8, 'trainlm');
net.trainParam.showWindow = false;
[net, tr] = train(net, X_train_n, y_train_n);

y_cap_mlp_train = mapminmax('reverse', net(X_train_n)', settings_y);
y_cap_mlp_val = mapminmax('reverse', net(X_val_n)', settings_y);

% Calcolo RMSE
RMSE_mlp_train = sqrt(mean((data_train.LOAD - y_cap_mlp_train).^2))
RMSE_mlp_val = sqrt(mean((data_val.LOAD - y_cap_mlp_val).^2))
%% surfacing con MLP a 8 neuroni

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23.9, 100); 
[X, Y] = meshgrid(x, y);

T = 24;
Y_sin = sin(2 * pi * Y(:) / T);
Y_cos = cos(2 * pi * Y(:) / T);

phi_vec = [X(:), Y_sin, Y_cos]'; 

phi_vec_n = mapminmax('apply', phi_vec, settings_x);

z_cap_n = net(phi_vec_n);
z_cap = mapminmax('reverse', z_cap_n, settings_y); 

Z = reshape(z_cap, size(X));

figure(22)
mesh(X, Y, Z)
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
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
y1 = mapminmax('reverse', net(input1_n)', settings_y);

% 00:01 (Appena dopo mezzanotte)
t2 = 0.01;
input2 = [temp_test, sin(2*pi*t2/24), cos(2*pi*t2/24)]';
input2_n = mapminmax('apply', input2, settings_x);
y2 = mapminmax('reverse',  net(input2_n)', settings_y);

fprintf('Carico alle 23:59: %.4f MW\n', y1);
fprintf('Carico alle 00:01: %.4f MW\n', y2);
fprintf('Salto di continuità: %.4f MW\n', abs(y1-y2));


