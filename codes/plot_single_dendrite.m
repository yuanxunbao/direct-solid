set(0,'defaultaxesfontsize',16)

dir = '../../tex/figures/';
growth_type = 'cell';

% load data
% load dirsolid_gpu_noise0.00E+00_misori0_lx180.0_nx128_asp10_seed720.mat
load dirsolid_cpu_noise2.00E-02_misori0_lx180.0_nx128_asp10_seed313.mat
% load dirsolid_cpu_noise2.00E-02_misori0_lx180.0_nx128_asp10_seed856.mat
% load dirsolid_cpu_noise4.00E-02_misori0_lx180.0_nx128_asp10_seed575.mat
% load dirsolid_gpu_noise2.00E-02_misori0_lx180.0_nx128_asp10_seed417.mat
% load dirsolid_cpu_noise0.00E+00_misori0_lx180.0_nx128_asp10_seed565.mat

% snapshot index
sz = size(order_param);
idx = 1:2:sz(2);
t_list = linspace(0,Tend,sz(2));


%%
% plot order parameter
figure(3);
set(gcf,'Position',[100,100,800,800])
for ss = 1 : length(idx)
    
    
    phi = order_param(:,ss); phi_r = reshape(phi, [nx,nz]);
    
    
    subplot(1, length(idx), ss) 
    surf(xx', zz', phi_r') ; shading interp; view(2); axis equal; axis tight
    xlabel('$x/W_0$', 'Interpreter','latex')
    ylabel('$z/W_0$', 'Interpreter','latex')
    title(sprintf('t = %.2f', t_list(idx(ss))) )
    
    colormap('default')
    cbar = colorbar('southoutside');
    cbar.Label.String = '$\phi$';
    cbar.Label.Interpreter = 'latex';
    cbar.Label.FontSize = 16;
    
    
    
end 

% print('-dpng',sprintf('%s/%s_phi.png',dir, growth_type),'-r300')


%%
% plot concentration field normalized by c_{inf}
figure(4);
set(gcf,'Position',[100,100,800,800])
for ss = 1 : length(idx)
    
    c = conc(:,ss); conc_r = reshape(c, [nx,nz]);
   

    subplot(1, length(idx), ss) 
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