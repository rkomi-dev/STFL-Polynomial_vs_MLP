clear 
clc
close all

%% fitting di modelli polinomiali ai minimi quadrati del carico elettrico in funzione di temp_media e ora del giorno

load('preprocessed_data.mat');
n = length(data_train.LOAD);
n_v = length(data_test.LOAD);
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

%% cross-validazione

n_v = length(data_test.LOAD);

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

min_SSR_V = min([SSR_V_quadratico, SSR_V_cubico, SSR_V_cubico_plus]);

fprintf('\nCROSS-VALIDAZIONE\n')
if(min_SSR_V == SSR_V_quadratico)
    disp('scelgo modello quadratico')
else 
    if(min_SSR_V == SSR_V_cubico)
        disp('scelgo modello cubico')
else
    disp('scelgo modello cubico plus')
    end
end