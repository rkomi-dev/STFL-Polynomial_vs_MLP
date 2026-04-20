# STFL: Polynomial vs MLP

## Data exploration

<img width="1728" height="1152" alt="carnet_plot" src="https://github.com/user-attachments/assets/41f62cd7-c31c-4902-9270-94f93ce77802" />

## Modelling with polynomial regression with different degrees 

* **Model complexity vs RMSE**

<img width="2160" height="1259" alt="complessità_poli" src="https://github.com/user-attachments/assets/fc3eb9cb-d213-47b8-b437-796ce728fb87" />

<br>

* **Response surface: 5th degree vs 5th degree + Fourier terms**

<img width="2160" height="1259" alt="confronto_sup_poli" src="https://github.com/user-attachments/assets/5eb5491d-5fd1-4d26-94f3-d030742a46ec" />


<br>

## Modelling with Stepwise regression method

* **Response surface: 5th degree + Fourier terms vs stepwise + Fourier terms**

<img width="2160" height="1259" alt="confronto_sup_poli_vs_step" src="https://github.com/user-attachments/assets/8ff12e19-8d3d-46aa-9eef-11f1888a9aa9" />


## Final model: stepwise + Fourier terms vs 8-neuron MLP

* **Neuron selection using k-fold cv and 1-SD rule**
 <img width="840" height="630" alt="scelta_neuroni" src="https://github.com/user-attachments/assets/c27a5f99-82e4-42c1-862c-8c10c4aec79a" />
<br>

* **Response surface and performance**

<img width="2160" height="1259" alt="confronto_sup_step_vs_mlp" src="https://github.com/user-attachments/assets/ebf65e33-6436-4b2b-b511-be826885447d" />

 <br>
 
* **Goodness of Fit**

<img width="1800" height="750" alt="confronto_gof_step_vs_mlp" src="https://github.com/user-attachments/assets/5cbb1f2f-8ee8-407f-af3b-c60d3649ff9a" />
