clear
clc
close all
%% Caricamento e Pre-processing
data = readtable('L1_train.csv');
data = rmmissing(data);

n_train = floor(0.70 * height(data));
data_train = data(1:n_train, :);         
data_test  = data(n_train+1:end, :);

temp_matrix = table2array(data(:, 3:27));
w_avg = mean(temp_matrix, 2);
w_avg_train = w_avg(1:n_train);
w_avg_test = w_avg(n_train+1:end);

%% Plot Esplorativi

%carico in funzione del tempo
figure(1)
plot(data.LOAD), grid on
xlabel('tempo'), ylabel('carico elettrico')
title('carico elettrico in funzione del tempo')

%temp_media in funzione del tempo 
figure(2)
plot(w_avg), grid on
xlabel('tempo'), ylabel('temperatura media') 
title('temperatura media in funzione del tempo')

% carico in funzione della temperatura media
figure(3)
scatter(w_avg, data.LOAD, '.'), grid on;
xlabel('temperatura media'), ylabel('carico elettrico')
title('carico elettrico in funzione della temperatura media')

%% Matrice di Correlazione
matrix_corr = corr(table2array(data(:, 2:27)));

% Salvataggio per gli script successivi
save('preprocessed_data.mat', 'w_avg', 'data_train', 'data_test', 'w_avg_train', 'w_avg_test');