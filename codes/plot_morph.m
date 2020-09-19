clear;


set(0,'defaultaxesfontsize',16)
set(0,'defaultlinelinewidth',2)

load ./ALCU_GRrun/dirsolid_varGR_traj1_noise0.04_misori0_lx90.50_nx666_asp5_ictype1_U0-1.00seed682.mat

sz = size(order_param);
t_list = linspace(0,Tend,sz(2));
idx = [10:2:21]; % frames to plot

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
% print('-depsc',sprintf('%s/%s_conc.eps',dir, growth_type),'-r300')

