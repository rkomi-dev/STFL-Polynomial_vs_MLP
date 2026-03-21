clear
clc
close all

%% caricamento dati
load('preprocessed_data.mat');

%% fitting di modelli polinomiali ai minimi quadrati del carico elettrico in funzione di temp_media

n = length(data_train.LOAD);
n_v = length(data_test.LOAD);
T_grid = (min(w_avg):0.1:max(w_avg))';

%% modello quadratico

q_quadratico = 3;

phi_quadratico_train = [ones(n, 1), w_avg_train, w_avg_train.^2];
[thetaLS_quadratico, std_error_quadratico, var_cap_quadratico, var_thetaLS_quadratico] = lscov(phi_quadratico_train, data_train.LOAD);

load_cap_quadratico = phi_quadratico_train * thetaLS_quadratico;
epsilon_quadratico = data_train.LOAD - load_cap_quadratico;
SSR_quadratico = epsilon_quadratico' * epsilon_quadratico;

%% fitting modello quadratico

Phi_grid = [ones(length(T_grid), 1) T_grid, T_grid.^2];
curva = Phi_grid*thetaLS_quadratico;

figure(6)
scatter(w_avg_train, data_train.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati training', 'Modello Quadratico')
title('Fitting: Carico vs Temperatura Media (training)')

figure(7)
scatter(w_avg_test, data_test.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati test', 'Modello Quadratico')
title('Fitting: Carico vs Temperatura Media (validazione)')

%% modello cubico

q_cubico = 4;

phi_cubico_train = [ones(n, 1), w_avg_train, w_avg_train.^2, w_avg_train.^3];
[thetaLS_cubico, std_error_cubico, var_cap_cubico, var_thetaLS_cubico] = lscov(phi_cubico_train, data_train.LOAD);

load_cap_cubico = phi_cubico_train * thetaLS_cubico;
epsilon_cubico = data_train.LOAD - load_cap_cubico;
SSR_cubico = epsilon_cubico' * epsilon_cubico;
%% fitting modello cubico

Phi_grid = [ones(length(T_grid), 1), T_grid, T_grid.^2, T_grid.^3];
curva = Phi_grid*thetaLS_cubico;

figure(8)
scatter(w_avg_train, data_train.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati training', 'Modello Cubico')
title('Fitting: Carico vs Temperatura Media (training)')

figure(9)
scatter(w_avg_test, data_test.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati test', 'Modello Cubico')
title('Fitting: Carico vs Temperatura Media (validazione)')

%% modello quarto grado

q = 5;

phi_quarto_train = [ones(n, 1), w_avg_train, w_avg_train.^2, w_avg_train.^3, w_avg_train.^4];
[thetaLS_quarto, std_error_quarto, var_cap_quarto, var_thetaLS_quarto] = lscov(phi_quarto_train, data_train.LOAD);


%% fitting modello quarto grado

Phi_grid = [ones(length(T_grid), 1), T_grid, T_grid.^2, T_grid.^3, T_grid.^4];
curva = Phi_grid*thetaLS_quarto;

figure(10)
scatter(w_avg_train, data_train.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati training', 'Modello quarto grado')
title('Fitting: Carico vs Temperatura Media (training)')

figure(11)
scatter(w_avg_test, data_test.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati test', 'Modello quarto grado')
title('Fitting: Carico vs Temperatura Media (validazione)')

%% test F

alpha = 0.05;

%% quadratico vs cubico

f_alpha = finv(1 - alpha, 1, n - q_quadratico);
f = (n - q_quadratico) * ((SSR_quadratico - SSR_cubico ) / SSR_cubico);
fprintf('\nTEST F\n')
fprintf('\nquadratico vs cubico: \n');
if(f < f_alpha) 
    disp('scelgo modello quadratico')
else
    disp('scelgo modello cubico')
end

%% cross-validazione

n_t = length(data_test.LOAD);

phi_V_quadratico = [ones(n_t, 1), w_avg_test, w_avg_test.^2];
load_cap_V_quadratico = phi_V_quadratico * thetaLS_quadratico;
epsilon_V_quadratico = data_test.LOAD - load_cap_V_quadratico;
SSR_V_quadratico = epsilon_V_quadratico' * epsilon_V_quadratico;

phi_V_cubico = [ones(n_t, 1), w_avg_test, w_avg_test.^2, w_avg_test.^3];
load_cap_V_cubico = phi_V_cubico * thetaLS_cubico;
epsilon_V_cubico = data_test.LOAD - load_cap_V_cubico;
SSR_V_cubico = epsilon_V_cubico' * epsilon_V_cubico;

fprintf('cross-validazione\n')
if(SSR_V_quadratico < SSR_V_cubico)
    disp('scelgo modello quadratico')
else
    disp('scelgo modello cubico')
end

%% considerazioni

% modello quarto grado scartato a prescindere dopo il plot perchè si vede
% che non può essere migliore