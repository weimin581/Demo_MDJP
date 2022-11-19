function [reconstructed_image_all , All_PSNR]= Inpainting_test_Demo(y,O,para,p,delta)

load Omega.mat;
P         =   Omega;

% 自定义的三个迭代步长
mu_A      =   para.mu_A;
mu_B      =   para.mu_B;
mu_C      =   para.mu_C;

IterNums  =   para.IterNums;
x_org     =   para.org;

All_PSNR  =   zeros(IterNums,1);
All_SSIM  =   zeros(IterNums,1);

iter = zeros(IterNums,1);

%初始化三个拉格朗日乘子
b = zeros(size(y));
c = zeros(size(y));
d = zeros(size(y));

x = para.Initial;

% imshow([x/255, y/255, O],[0,1]);

Opts.PatchSize = para.patch;
tic

for j = 1 : IterNums

    % 计算每一次迭代的中间精度
    All_PSNR(j) = csnr(x,x_org,0,0);
    All_SSIM(j) = ssim(x,x_org);
    
    iter(j) = j;
    
    % 第一个约束： X = Z
    % (WSNM，这个是baseline model, fixed)：在New_method中，这块是不变的！！！
    z = Solve_Z_GST(x - b,para,p); 

    % 局部模型部分：
    % 将第二个约束： GSR(创新点1：GSR_2014_TIP_张健) ：
    w = GSR_Solver_Inpainting(x - c, Opts);
   
    % 第三个约束：
    % 参数设为input image noise level = delta
    % 此处可以理解为：利用FFDNet进行单张灰度图的去噪！！！
    s = FFD_Net_Denoiser(x - d, delta);
    
    % 总的计算公式（里面的y的初始值：实际的degraded image）
    x = (y + mu_A * (z + b)+ mu_B * (w + c) + mu_C * (s+d))./(O+(mu_A + mu_B + mu_C) * ones(size(O)));
 
    
    % 三个约束对应的三个拉格朗日乘子(B,C,D)的更新公式
    b = b-(x-z); % 乘子b跟约束z绑定
    c = c-(x-w); % 乘子c跟约束w绑定
    d = d-(x-s); % 乘子d跟约束s绑定

    mu_A = 1.01 * mu_A;
    mu_B = 1.01 * mu_B;
    mu_C = 1.01 * mu_C;

    % 迭代停止条件
    if j>1
        if(All_PSNR(j)-All_PSNR(j-1)<0)
               fprintf('最优结果：iter number = %d, PSNR = %f, SSIM = %f\n',(j-1), All_PSNR(j-1),All_SSIM(j-1));
            break;
        end
    end


       fprintf('x: iter number = %d, PSNR = %f, SSIM = %f\n', j,csnr(x,x_org,0,0),ssim(x,x_org));
end
toc

x(x>255) = 255;
x(x<0) = 0;   

reconstructed_image_all = x;

end

