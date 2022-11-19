function [reconstructed_image_all ,  best_psnr , best_ssim]= MDJP_Solved_by_AM_ADMM(y,O,para,p,delta,rou)

mu =  0.0008;
mu_A      =   para.mu_A;
mu_B      =   para.mu_B;

IterNums  =   para.IterNums;
x_org     =   para.org;

All_PSNR  =   zeros(IterNums,1);
All_SSIM  =   zeros(IterNums,1);

iter = zeros(IterNums,1);

b = zeros(size(y));
c = zeros(size(y));

x = para.Initial;

Opts.PatchSize = para.patch;

tic

for j = 1 : IterNums

    All_PSNR(j) = csnr(x,x_org,0,0);
    All_SSIM(j) = ssim(x,x_org);
    
    iter(j) = j;

    Prior1 = Solve_GSR( x , Opts);   
    
    Prior2 = Solve_Z_GST( x - b , para , p );    
    
    Prior3 = FFD_Net_Denoiser(x - c , delta); 
  
    x = (y +  mu * Prior1 +  mu_A * ( Prior2 + b )+ mu_B * ( Prior3 + c ) ) ./ (O + ( mu_A + mu_B ) * ones( size(O)) + mu );
    
    b = b-(x-Prior2); 
    c = c-(x-Prior3); 
    
    mu_A = rou * mu_A;
    mu_B = rou * mu_B;
    
    if j>1
        if(All_PSNR(j)-All_PSNR(j-1)<0)
               fprintf('final res£ºnoise_factor = %d, iter number = %d, PSNR = %f, SSIM = %f\n',delta,(j-1), All_PSNR(j-1),All_SSIM(j-1));
               best_psnr = All_PSNR(j-1);
               best_ssim = All_SSIM(j-1);
            break;
        end
    end
    
    fprintf('x: iter number = %d, PSNR = %f, SSIM = %f\n', j,csnr(x,x_org,0,0),ssim(x,x_org));

end

toc

x(x>255) = 255;
x(x<0) = 0;   

% final result for reconstructing 
reconstructed_image_all = x;

end

