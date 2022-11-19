function  x_final = Inpainting_JSM(y,Options)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function solves the following problem with compound regularization
%
%     arg min_x = 0.5*|| y - A x ||_2^2 + lambda*Psi_LSM( x ) + tau*Psi_NLSM( x )
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mask = Options.A;
A = @(x) mask.*x;
AT = @(x) mask.*x;

ATy = AT(y);

mu = Options.mu;
mu1 = 0.14*mu; % Parameter for LSM
mu2 = 0.86*mu; % Parameter for NLSM

BlockSize = Options.BlockSize;

x = Options.initial;
IterNums = Options.IterNums;
true = Options.true;

% All_PSNR  =   zeros(IterNums,1);
% imshow([y/255, true/255, x/255],[0,1]);

b = zeros(size(y));
c = zeros(size(y));
w = zeros(size(y));

muinv = 1/mu;
TVIterNum = 20;
invAAT = 1./(mu+mask);
InIterNums = 1;
lambda = 5;
% MSE = zeros(1,IterNums+1);
% MSE(1) = sum(sum((x-true).^2))/numel(x);

% fprintf('Degraded PSNR = %0.2f\n',csnr(y,true,0,0));
% fprintf('Initial PSNR = %0.2f\n',csnr(x,true,0,0));

Opts.PatchSize = BlockSize;


for Outloop = 1:IterNums
    
    for Inloop = 1:InIterNums
        
        z = Solve_LSM(x-b,lambda,TVIterNum);
        
        w = New_Solve_NLSM(x-c,Opts);
%          r = ATy +mu2*(w+c);
        r = ATy + mu1*(z+b)+mu2*(w+c);
        
        x = muinv*(r - AT(invAAT.*A(r)));
        
    end
    
%     b = b + (z - x);
    
    c = c + (w - x);
    
%     x_resid = x - true;
%     MSE(Outloop+1) =  (x_resid(:)'*x_resid(:))/numel(x);
%     fprintf('iter number = %d, PSNR = %0.2f\n',Outloop,csnr(x,true,0,0));

    fprintf('iter number = %d, PSNR = %f, ssim = %f\n',Outloop,csnr(x,true,0,0),ssim(x,true));


end

x_final = w;

fprintf('Final PSNR = %f\n',csnr(x_final,true,0,0));

end

