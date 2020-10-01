clear;close all;


set(0,'defaultaxesfontsize',16)
set(0,'defaultlinelinewidth',2)

%load ./ALCU_GRrun/dirsolid_varGR_traj1_noise0.04_misori0_lx90.50_nx666_asp5_ictype1_U0-1.00seed682.mat
load dirsolid_varGR_traj1_noise0.00_misori0_lx18.10_nx133_asp40_ictype1_U0-1.00seed822.mat
load dirsolid_varGR_traj1_noise0.00_misori0_lx18.10_nx133_asp40_ictype1_U0-1.00seed822_QoIs.mat
sz = size(order_param);
t_trans= 0.1;
t_list = linspace(0,Tend,sz(2))-t_trans;
idx = [5:4:21]; % frames to plot
idx = [1:4:21]
for ss = 1 : length(idx)
    
    cc = conc(:,idx(ss)); phi = reshape(cc, [nx,nz]);
    z= zz_mv(:,idx(ss));
    x = xx(:,2);
    
    [zz,xx]=meshgrid(z,x);
    
    figure(3)
    subplot(1, length(idx), ss) 
    surf(xx', zz', phi') ; shading interp; view(2); axis equal; axis tight
    xlabel('$x$ $(\mu m)$', 'Interpreter','latex')
    ylabel('$z$ $(\mu m)$', 'Interpreter','latex')
    title(sprintf('t = %.2e', t_list(idx(ss))) )
    % axis([0,1,28.9,29.9])
    
    colormap('jet')
    caxis([0,5])
    cbar = colorbar('southoutside');
    cbar.Label.String = '$c/c_{\infty}$';
    cbar.Label.Interpreter = 'latex';
    cbar.Label.FontSize = 16;
    
    
    figure(5);
    plot(phi(4,:)); hold on;
    
    
end 
figure(3)
%print('-depsc',sprintf('%s/%s_conc.eps',dir, growth_type),'-r300')

%trun_len = 3331;
trun_st = 1200;
trun_end = nz;%trun_st + trun_len -1;

psi_last = reshape(order_param(:,end), [nx,nz]);
c_last = reshape(conc(:,end), [nx,nz]);
U_last = reshape(Uc(:,end), [nx,nz]);
ztip = ztip_qoi(end);
zz_sh = zz;

psi_1d = psi_last(2,trun_st:trun_end);
c_1d = c_last(2,trun_st:trun_end);
z_1d = zz_sh(2,trun_st:trun_end);
U_1d = U_last(2,trun_st:trun_end);
Ttip = Ttip_arr(end);

phi_1d = tanh(psi_1d/sqrt(2));

figure(6)
subplot(221)
plot(phi_1d);
subplot(222)
plot(c_1d);
subplot(223)
plot(z_1d);
subplot(224)
plot(U_1d);

%save transient1d.mat psi_1d U_1d z_1d Ttip













