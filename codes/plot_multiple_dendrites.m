% load dirsolid_noise4.00E-02_misori0_lx3600.0_nx2560_asp0.5_U0-3.00E-01_seed663.mat

% load dirsolid_gpu_noise4.00E-02_misori0_lx2812.5_nx2000_asp0.5_seed192.mat
load dirsolid_gpu_noise4.00E-02_misori15_lx2812.5_nx2000_asp0.5_seed105.mat
% load dirsolid_gpu_noise4.00E-02_misori30_lx2812.5_nx2000_asp0.5_seed198.mat
% load dirsolid_gpu_noise4.00E-02_misori45_lx2812.5_nx2000_asp0.5_seed613.mat

dir = '../../tex/figures/';
growth_type = 'multi_dendrite_misori45';

% snapshot index
sz = size(order_param);
idx = 1:2:sz(2);
t_list = linspace(0,Tend,sz(2));



figure(2); 
for ss = 1 : length(idx)
    
    c = conc(:,ss); conc_r = reshape(c, [nx,nz]);
   
    figure(2)
    subplot(3, ceil(length(idx)/3), ss) 
    surf(xx', zz', conc_r') ; shading interp; view(2); axis equal; axis tight
    xlabel('$x/W_0$', 'Interpreter','latex')
    ylabel('$z/W_0$', 'Interpreter','latex')
    title(sprintf('t = %.2f', t_list(idx(ss))) )
    
    colormap('jet')
    caxis([0,4])
    cbar = colorbar('southoutside');
    cbar.Label.String = '$c/c_{\infty}$';
    cbar.Label.Interpreter = 'latex';
    cbar.Label.FontSize = 16;
    
end 


% print('-dpng',sprintf('%s/%s_conc.png',dir, growth_type),'-r300')
