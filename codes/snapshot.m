

load dendrite.mat

[nv,nt]=size(y); ntp = (nt-1)/5;

nt1 = ntp+1;nt2 = 2*ntp+1;nt3 = 3*ntp+1;
nt4 = 4*ntp+1; 


xx3=xx;zz3=zz;
figure(5)
subplot(1,6,1);
surf(xx3,zz3,reshape((y(1:nz*nx,1)),[nz,nx]));shading interp;
view(2); axis tight;axis equal; 
tit1=strcat('t=',num2str(0));
title(tit1)
subplot(1,6,2);
surf(xx3,zz3,reshape((y(1:nz*nx,nt1)),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit2=strcat('t=',num2str((nt4-1)/(nt-1)*t));
title(tit2)
subplot(1,6,3);
surf(xx3,zz3,reshape((y(1:nz*nx,nt2)),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit3=strcat('t=',num2str((nt2-1)/(nt-1)*t));
title(tit3)
subplot(1,6,4);
surf(xx3,zz3,reshape((y(1:nz*nx,nt3)),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit4=strcat('t=',num2str((nt3-1)/(nt-1)*t));
title(tit4)
subplot(1,6,5);
surf(xx3,zz3,reshape((y(1:nz*nx,nt4)),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit5=strcat('t=',num2str((nt4-1)/(nt-1)*t));
title(tit5)
subplot(1,6,6);
surf(xx3,zz3,reshape((y(1:nz*nx,nt)),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit6=strcat('t=',num2str(t));
title(tit6)


figure(6)
subplot(1,6,1);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,1),[nz,nx]));shading interp;
view(2); axis tight;axis equal; colormap('jet');
tit1=strcat('t=',num2str(0));
title(tit1)
subplot(1,6,2);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,nt1),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; colormap('jet');
tit2=strcat('t=',num2str((nt1-1)/(nt-1)*t));
title(tit2)
subplot(1,6,3);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,nt2),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; colormap('jet');
tit3=strcat('t=',num2str((nt2-1)/(nt-1)*t));
title(tit3)
subplot(1,6,4);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,nt3),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; colormap('jet');
tit4=strcat('t=',num2str((nt3-1)/(nt-1)*t));
title(tit4)
subplot(1,6,5);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,nt4),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; colormap('jet');
tit5=strcat('t=',num2str((nt4-1)/(nt-1)*t));
title(tit5)
subplot(1,6,6);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,nt),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; colormap('jet');
tit6=strcat('t=',num2str(t));
title(tit6)






