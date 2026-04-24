# STFL: Polynomial vs MLP

## Data exploration

<img width="1728" height="1152" alt="carnet_plot" src="https://github.com/user-attachments/assets/41f62cd7-c31c-4902-9270-94f93ce77802" />

## Modelling with polynomial regression with different degrees 

* **Model complexity vs RMSE**

<img width="2160" height="1259" alt="complessità_poli" src="https://github.com/user-attachments/assets/fc3eb9cb-d213-47b8-b437-796ce728fb87" />

<br>

* **Response surface: 5th degree vs 5th degree + Fourier terms**

<img width="2160" height="1259" alt="confronto_sup_poli" src="https://github.com/user-attachments/assets/e2e9f5a6-6acc-4682-a2e0-8e62dfcee0a6" />



<br>

## Modelling with Stepwise regression method

* **Response surface: 5th degree + Fourier terms vs stepwise + Fourier terms**

<img width="2160" height="1259" alt="confronto_sup_poli_vs_step" src="https://github.com/user-attachments/assets/12b4683b-2e00-4ffd-97eb-e2f3a315be3b" />



## Final comparison using average temperature: Stepwise + Fourier vs 8-neuron MLP

* **Neuron selection using k-fold cv and 1-SD rule**
 <img width="840" height="630" alt="scelta_neuroni" src="https://github.com/user-attachments/assets/c27a5f99-82e4-42c1-862c-8c10c4aec79a" />
<br>

* **Response surface and performance**

<img width="2160" height="1259" alt="confronto_sup_step_vs_mlp" src="https://github.com/user-attachments/assets/f55d4d3e-0a03-4c6e-8a51-ed3a4a159beb" />


 <br>
 
* **Goodness of Fit**

<img width="2160" height="1259" alt="confronto_gof_step_vs_mlp" src="https://github.com/user-attachments/assets/414e4394-e2c1-4e98-950b-3f115857d9fb" />


<br>

* **Dispersion of residues**

<img width="2160" height="1259" alt="disp_residui" src="https://github.com/user-attachments/assets/e1901be4-5ba4-4f04-9e80-cbde3f17984a" />

<br>

* **Residue histogram**

<img width="2160" height="1259" alt="isto_residui" src="https://github.com/user-attachments/assets/07198afa-75d7-4be4-b998-46b633d1c664" />

## Final model using all 25 temperatures


* **Goodness of Fit**

<img width="2160" height="1259" alt="confronto_gof_mlp" src="https://github.com/user-attachments/assets/272778be-a9e4-45a3-8405-b438455a05e4" />

<br>

* **Prediction**

<img width="2160" height="1259" alt="confronto_pred_mlp" src="https://github.com/user-attachments/assets/a8ef2ec3-9429-4740-a90c-9ee811c9a3ed" />

## Performance of all models

| MODELLO | RMSE | MAPE | R2 |
| :--- | ---: | ---: | ---: |
| Polinomio 5° grado + Fourier | 17.93 | 10.14% | 0.8507 |
| stepwise + Fourier | 15.96 | 8.75% | 0.8818 |
| mlp con 8 neuroni | 15.06 | 8.24% | 0.8947 |
| mlp con 8 neuroni (25 temp) | 11.17 | 6.29% | 0.9420 |
| mlp con 19 neuroni (25 temp) | 10.46 | 5.88% | 0.9492 |
