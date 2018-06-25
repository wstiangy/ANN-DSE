clear all;
close all;
clc;

filename = '9bus_data';

[times1,dt,Vre,Vim,Vang_mes,mac_ang,mac_spd,RV_p,Param,U_MM_sq,U_MM,time_srukf,time_ukf]= data_prep_Vre_Vim(filename);

t_axis  = 0:times1/(length(mac_spd(1,:))-1):times1;
epoch   = length(t_axis); 
wB = 2*pi*60;           % base angular speed
Et = Param(:,3);

%% ============ KALMAN INITIALIZATION =================
std_dev  = 1e-3; 
var = std_dev^2;
even = 2:2:6;  odd  = 1:2:6;

% Measurement Noise Covariances 
V  = eye(6);
R  = eye(6)*var;

% Process Noise Covariances
W = 0.75e-5*eye(6,6);
Q = eye(6,6);

% Initial Conditions 
D_est_KAL  = zeros(3,1);
W_est_KAL  = ones(3,1);
P = ones(6,6);

D_prev_KAL = zeros(3,1);
W_prev_KAL = ones(3,1);


% Initial States 
W_est_arr_EKF = [];
D_est_arr_EKF = []; 
vis_cov=[];

tstart_ekf = tic;
for k = 1:epoch-1
    % Measurements        
        Vmes  = Vre(:,k)+1i*Vim(:,k);
        Vmes_next  = Vre(:,k+1)+1i*Vim(:,k+1);        
        
    % ==================== EXTENDED KALMAN FILTER =========================   
        % Calculate Jacobian Matrices 
        [A,Hk]                  = jacobians(D_est_KAL,D_prev_KAL,Vmes,Param,RV_p);
        
        %  Model Based Prediction
        [D_prev_KAL,W_prev_KAL] = swing_model(D_est_KAL,W_est_KAL,Vmes,Param,dt);
        Xprev                   = alternate(D_prev_KAL,W_prev_KAL);

        A          = (eye(6)+ (A*dt));
        Pk_prev    = (A*P*A') + (W*Q*W');
        K_GAIN     = (Pk_prev*Hk')*inv((Hk*Pk_prev*Hk') + (V*R*V'));
        
        % Predict Measurement
        [x,y]       = pol2cart(D_prev_KAL,Et);  Eterm   = x+1i*y;             
        Vmes_pred   = (RV_p*Eterm);

        zk_hat      = alternate(real(Vmes_pred),imag(Vmes_pred));    
        zk	        = alternate(real(Vmes_next),imag(Vmes_next));

        X_post      = Xprev + (K_GAIN*(zk - zk_hat));

        D_est_KAL   = X_post(odd);    
        W_est_KAL   = X_post(even);

        P           = (eye(6) - (K_GAIN*Hk))*Pk_prev;
           
        
    % ====================== STORE ARRAYS ==============================
        W_est_arr_EKF   = [W_est_arr_EKF W_est_KAL];
        D_est_arr_EKF   = [D_est_arr_EKF D_est_KAL];    
        
end
time_ekf = toc(tstart_ekf);

%% ============ NEURAL-ALGORITHM INITIALIZATIONS =================
D_est_arr_NNE = [];
W_est_arr_NNE = [];

gamma1 = 0.1;          % learning rate delta
eta1 = 0.1;                   % delta
alpha1 = 1;

% gamma2 = 0.005;          % learning rate omega
eta2 = 0.01;                   % omega
alpha2 = 1;

% initialize NN

load('net_init_9bus')

d_err_acc = 0;
w_err_acc = 0;

D_1next=mac_ang(:,1);

% ===================== ITERATIVE STATE ESTIMATION =======================
tstart_nn = tic;
for k = 1:epoch-1
    % =====================Genearate Measurements =========================        
        % Current and next measurement 
        Vang = Vang_mes(:,k);
        Vang_next = Vang_mes(:,k+1);
        
        D_last=D_1next;

    % ====================== NEURAL NETWORK ==============================
    
        PMU_mes1 = [Vre(:,k);Vim(:,k)];
        [D_1next,net1] = BNN_forward(PMU_mes1,net1);
        
        PMU_mes2 = alternate(D_1next,Vang);        
        [W,net2] = BNN_forward(PMU_mes2,net2);  
        
        % Model Based Prediction
        [D_2next,W_next,omega_dt] = swing_modelpol(D_last,W,Vang,Param,dt);
        
        [x,y]           = pol2cart(D_1next,Et);  
        Eterm           = x+1i*y;  % Terminal Voltage at k+1

        % Predict Measurements at k+1
        V_pred_BNN = (RV_p*Eterm);    
        V_pred_BNN_theta=angle(V_pred_BNN);
        
        % Error Calculation
        d_err(:,k)   = wrapToPi(V_pred_BNN_theta - Vang_next);
        w_err(:,k)   = (D_2next - D_1next);
        
        % dynamic learning rate
        gamma2(k) = 0.04/(1+exp(k/120-2))+0.01;
        
        % Adjust Neural Network Weights 
            [net1]  = BNN_backward(net1,d_err(:,k),gamma1);  % delta
            [net2]  = BNN_backward(net2,w_err(:,k),gamma2(k));  % omega
    % ====================== STORE ARRAYS ==============================
        D_est_arr_NNE   = [D_est_arr_NNE D_1next];
        W_est_arr_NNE   = [W_est_arr_NNE W];
end
time_nn = toc(tstart_nn);

%% ========================= DISPLAY =========================
% Truncate
range_plt = 1:10*120;
t = t_axis(range_plt);
D_est_arr_EKF = D_est_arr_EKF(:,range_plt);
W_est_arr_EKF = W_est_arr_EKF(:,range_plt);
D_est_arr_NNE = D_est_arr_NNE(:,range_plt);
W_est_arr_NNE = W_est_arr_NNE(:,range_plt);
D_est_arr_UKF = U_MM(1:3,range_plt);
W_est_arr_UKF = U_MM(4:6,range_plt);
D_est_arr_SR = U_MM_sq(1:3,range_plt);
W_est_arr_SR = U_MM_sq(4:6,range_plt);

number = size(range_plt,2)*3;
% EKF error
D_est_err_EKF = sqrt(sum(sum((mac_ang(:,range_plt)-D_est_arr_EKF).^2))/number);
W_est_err_EKF = sqrt(sum(sum((wB*mac_spd(:,range_plt,:)-wB*W_est_arr_EKF).^2))/number);

% NN error
D_est_err_NNE = sqrt(sum(sum((mac_ang(:,range_plt)-D_est_arr_NNE).^2))/number);
W_est_err_NNE = sqrt(sum(sum((wB*mac_spd(:,range_plt)-wB*W_est_arr_NNE).^2))/number);

% UKF error
D_est_err_UKF =sqrt(sum(sum((mac_ang(:,range_plt)-D_est_arr_UKF).^2))/number);
W_est_err_UKF =sqrt(sum(sum((wB*mac_spd(:,range_plt)-W_est_arr_UKF).^2))/number);

% SRUKF error
D_est_err_SR =sqrt(sum(sum((mac_ang(:,range_plt)-D_est_arr_SR).^2))/number);
W_est_err_SR =sqrt(sum(sum((wB*mac_spd(:,range_plt)-W_est_arr_SR).^2))/number);

% plot
for p=1:3
    figure
    y_ax_lim1 = [min(mac_ang(p,:))-0.01 max(mac_ang(p,:))+0.01];
    y_ax_lim2 = [min(wB*mac_spd(p,:))-2 max(wB*mac_spd(p,:))+2];
    subplot(2,1,1)
            hold on;
            plot(t,mac_ang(p,range_plt),'color','black','Linewidth',1.5); 
            plot(t,D_est_arr_NNE(p,range_plt),'r--','Linewidth',1.5);
            plot(t,D_est_arr_EKF(p,range_plt),'b--','Linewidth',1.5);
            plot(t,D_est_arr_UKF(p,range_plt),'g--','Linewidth',1.5);
            plot(t,D_est_arr_SR(p,range_plt),'m--','Linewidth',1.5);
            legend('Actual','NN','EKF','UKF','SR-UKF');
            xlabel('time(sec)');
            ylabel('Generator Angle (\delta) in p.u');
            ylim(y_ax_lim1);  
            xlim([0 10]);
            grid on;
            title(['\delta-' num2str(p) ''])
    subplot(2,1,2)
            hold on;
            plot(t,wB*mac_spd(p,range_plt),'color','black','Linewidth',1.5); 
            plot(t,wB*W_est_arr_NNE(p,range_plt),'r--','Linewidth',1.5);
            plot(t,wB*W_est_arr_EKF(p,range_plt),'b--','Linewidth',1.5);
            plot(t,W_est_arr_UKF(p,range_plt),'g--','Linewidth',1.5);
            plot(t,W_est_arr_SR(p,range_plt),'m--','Linewidth',1.5);
            legend('Actual','NN','EKF','UKF','SR-UKF');
            xlabel('time(sec)');
            ylabel('Angular Frequency (\omega) in p.u');
            ylim(y_ax_lim2); 
            xlim([0 10]);
            grid on;
            title(['\omega-' num2str(p) ''])
end

% Gather time and error array
time_array = [time_ekf;time_nn;time_ukf;time_srukf];
err_delta_array = [D_est_err_EKF;D_est_err_NNE;D_est_err_UKF;D_est_err_SR];
err_omega_array = [W_est_err_EKF;W_est_err_NNE;W_est_err_UKF;W_est_err_SR];


