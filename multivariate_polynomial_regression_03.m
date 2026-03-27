clear 
clc
close all

%% fitting di modelli polinomiali ai minimi quadrati del carico elettrico in funzione di temp_media e ora del giorno

load('preprocessed_data.mat');

%% scatter dei dati
figure(14)
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 'b'); xlabel('temp media'), ylabel('ora del giorno'), zlabel('LOAD'), grid on;
title('scatter identificazione');

figure(15)
scatter3(w_avg_test, data_test.TIMESTAMP, data_test.LOAD, '*', 'r');
title('scatter validazione');

%% modello quadratico + 1 termine di interazione

q_quadratico = 6;
phi_quadratico = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2, w_avg_train.*data_train.TIMESTAMP];

[thetaLS_quadratico, std_error_quadratico, var_cap_quadratico, var_thetaLS_quadratico] = lscov(phi_quadratico, data_train.LOAD);

load_cap_quadratico = phi_quadratico * thetaLS_quadratico;
epsilon_quadratico = data_train.LOAD - load_cap_quadratico;
SSR_quadratico = epsilon_quadratico' * epsilon_quadratico;

%% surfacing modello quadratico

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23, 24);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.*y_vec];
z_cap = phi_vec * thetaLS_quadratico;

Z = reshape(z_cap, size(X));
figure(16)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello quadratico')
%% modello cubico

q_cubico = 8;

phi_cubico = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2, w_avg_train.*data_train.TIMESTAMP, w_avg_train.^3, data_train.TIMESTAMP.^3];

[thetaLS_cubico, std_error_cubico, var_cap_cubico, var_thetaLS_cubico] = lscov(phi_cubico, data_train.LOAD);

load_cap_cubico = phi_cubico * thetaLS_cubico;
epsilon_cubico = data_train.LOAD - load_cap_cubico;
SSR_cubico = epsilon_cubico' * epsilon_cubico;
%% surfacing modello cubico

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23, 24);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.*y_vec, x_vec.^3, y_vec.^3];
z_cap = phi_vec * thetaLS_cubico;

Z = reshape(z_cap, size(X));
figure(17)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello cubico')
%% modello cubico + 2 termini di interazione

q_cubico_plus = 10;
phi_cubico_plus = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2, w_avg_train.*data_train.TIMESTAMP, w_avg_train.^3, data_train.TIMESTAMP.^3, (w_avg_train.^2).*data_train.TIMESTAMP, w_avg_train.*(data_train.TIMESTAMP.^2)];

[thetaLS_cubico_plus, std_error_cubico_plus, var_cap_cubico_plus, var_thetaLS_cubico_plus] = lscov(phi_cubico_plus, data_train.LOAD);

load_cap_cubico_plus = phi_cubico_plus * thetaLS_cubico_plus;
epsilon_cubico_plus = data_train.LOAD - load_cap_cubico_plus;
SSR_cubico_plus = epsilon_cubico_plus' * epsilon_cubico_plus;
%% surfacing modello cubico + 2 termini di interazione

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23, 24);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.*y_vec, x_vec.^3, y_vec.^3, (x_vec.^2).*y_vec, x_vec.*(y_vec.^2)];
z_cap = phi_vec * thetaLS_cubico_plus;

Z = reshape(z_cap, size(X));
figure(18)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello cubico plus')

%% modello quarto grado

q_quarto = 12;
phi_quarto = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2, w_avg_train.*data_train.TIMESTAMP, w_avg_train.^3, data_train.TIMESTAMP.^3, (w_avg_train.^2).*data_train.TIMESTAMP, w_avg_train.*(data_train.TIMESTAMP.^2), w_avg_train.^4, data_train.TIMESTAMP.^4];

[thetaLS_quarto, std_error_quarto, var_cap_quarto, var_thetaLS_quarto] = lscov(phi_quarto, data_train.LOAD);

load_cap_quarto = phi_quarto * thetaLS_quarto;
epsilon_quarto = data_train.LOAD - load_cap_quarto;
SSR_quarto = epsilon_quarto' * epsilon_quarto;
%% surfacing modello quarto grado

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 23, 24);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.*y_vec, x_vec.^3, y_vec.^3, (x_vec.^2).*y_vec, x_vec.*(y_vec.^2), x_vec.^4, y_vec.^4];
z_cap = phi_vec * thetaLS_quarto;

Z = reshape(z_cap, size(X));
figure(18)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello quarto grado')
%% cross-validazione

phi_V_quadratico = [ones(n_v, 1), w_avg_test, data_test.TIMESTAMP, w_avg_test.^2, data_test.TIMESTAMP.^2, w_avg_test.*data_test.TIMESTAMP];
load_cap_V_quadratico = phi_V_quadratico * thetaLS_quadratico;
epsilon_V_quadratico = data_test.LOAD - load_cap_V_quadratico;
SSR_V_quadratico = epsilon_V_quadratico' * epsilon_V_quadratico;

phi_V_cubico = [ones(n_v, 1), w_avg_test, data_test.TIMESTAMP, w_avg_test.^2, data_test.TIMESTAMP.^2, w_avg_test.*data_test.TIMESTAMP, w_avg_test.^3, data_test.TIMESTAMP.^3];
load_cap_V_cubico = phi_V_cubico * thetaLS_cubico;
epsilon_V_cubico = data_test.LOAD - load_cap_V_cubico;
SSR_V_cubico = epsilon_V_cubico' * epsilon_V_cubico;

phi_V_cubico_plus = [ones(n_v, 1), w_avg_test, data_test.TIMESTAMP, w_avg_test.^2, data_test.TIMESTAMP.^2, w_avg_test.*data_test.TIMESTAMP, w_avg_test.^3, data_test.TIMESTAMP.^3,  (w_avg_test.^2).*data_test.TIMESTAMP, w_avg_test.*(data_test.TIMESTAMP.^2)];
load_cap_V_cubico_plus = phi_V_cubico_plus * thetaLS_cubico_plus;
epsilon_V_cubico_plus = data_test.LOAD - load_cap_V_cubico_plus;
SSR_V_cubico_plus = epsilon_V_cubico_plus' * epsilon_V_cubico_plus;

phi_V_quarto = [ones(n_v, 1), w_avg_test, data_test.TIMESTAMP, w_avg_test.^2, data_test.TIMESTAMP.^2, w_avg_test.*data_test.TIMESTAMP, w_avg_test.^3, data_test.TIMESTAMP.^3,  (w_avg_test.^2).*data_test.TIMESTAMP, w_avg_test.*(data_test.TIMESTAMP.^2), w_avg_test.^4, data_test.TIMESTAMP.^4];
load_cap_V_quarto = phi_V_quarto * thetaLS_quarto;
epsilon_V_quarto = data_test.LOAD - load_cap_V_quarto;
SSR_V_quarto = epsilon_V_quarto' * epsilon_V_quarto;
min_SSR_V = min([SSR_V_quadratico, SSR_V_cubico, SSR_V_cubico_plus, SSR_V_quarto]);

fprintf('\nCROSS-VALIDAZIONE:\n')
if(min_SSR_V == SSR_V_quadratico)
    disp('scelgo modello quadratico')
else 
    if(min_SSR_V == SSR_V_cubico)
        disp('scelgo modello cubico')
else
    if(min_SSR_V == SSR_V_cubico_plus)
        disp('scelgo modello cubico plus')
    else
        disp('scelgo modello quarto grado')
    end
    end
end

%% confronto complessità vs RMSE

RMSE_quadratico_train = sqrt(SSR_quadratico / n)
RMSE_quadratico_test = sqrt(SSR_V_quadratico / n_v)
RMSE_cubico_train = sqrt(SSR_cubico / n)
RMSE_cubico_test = sqrt(SSR_V_cubico / n_v)
RMSE_cubico_plus_train = sqrt(SSR_cubico_plus / n)
RMSE_cubico_plus_test = sqrt(SSR_V_cubico_plus / n_v)
RMSE_quarto_train = sqrt(SSR_quarto / n)
RMSE_quarto_test = sqrt(SSR_V_quarto / n_v)


rmse_train_vals = [RMSE_quadratico_train, RMSE_cubico_train, RMSE_cubico_plus_train RMSE_quarto_train];
rmse_test_vals  = [RMSE_quadratico_test, RMSE_cubico_test, RMSE_cubico_plus_test, RMSE_quarto_test];
x_axis = 1:4; 

figure(19);
plot(x_axis, rmse_train_vals, '-o', 'MarkerFaceColor', 'b');
hold on
plot(x_axis, rmse_test_vals, '-s', 'MarkerFaceColor', 'r');

grid on
xticks(x_axis)
xticklabels({'Quadratico (q=6)', 'Cubico (q=8)', 'Cubico plus (q=10)', 'Quarto(q=12)'})
ylabel('RMSE (MW)')
xlabel('Complessità del Modello (Grado Polinomio)')
title('complessità vs RMSE')
legend('Train RMSE', 'Test RMSE', 'Location', 'northeast')
