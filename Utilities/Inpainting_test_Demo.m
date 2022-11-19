function [reconstructed_image_all , All_PSNR]= Inpainting_test_Demo(y,O,para,p,delta)

load Omega.mat;
P         =   Omega;

% �Զ����������������
mu_A      =   para.mu_A;
mu_B      =   para.mu_B;
mu_C      =   para.mu_C;

IterNums  =   para.IterNums;
x_org     =   para.org;

All_PSNR  =   zeros(IterNums,1);
All_SSIM  =   zeros(IterNums,1);

iter = zeros(IterNums,1);

%��ʼ�������������ճ���
b = zeros(size(y));
c = zeros(size(y));
d = zeros(size(y));

x = para.Initial;

% imshow([x/255, y/255, O],[0,1]);

Opts.PatchSize = para.patch;
tic

for j = 1 : IterNums

    % ����ÿһ�ε������м侫��
    All_PSNR(j) = csnr(x,x_org,0,0);
    All_SSIM(j) = ssim(x,x_org);
    
    iter(j) = j;
    
    % ��һ��Լ���� X = Z
    % (WSNM�������baseline model, fixed)����New_method�У�����ǲ���ģ�����
    z = Solve_Z_GST(x - b,para,p); 

    % �ֲ�ģ�Ͳ��֣�
    % ���ڶ���Լ���� GSR(���µ�1��GSR_2014_TIP_�Ž�) ��
    w = GSR_Solver_Inpainting(x - c, Opts);
   
    % ������Լ����
    % ������Ϊinput image noise level = delta
    % �˴��������Ϊ������FFDNet���е��ŻҶ�ͼ��ȥ�룡����
    s = FFD_Net_Denoiser(x - d, delta);
    
    % �ܵļ��㹫ʽ�������y�ĳ�ʼֵ��ʵ�ʵ�degraded image��
    x = (y + mu_A * (z + b)+ mu_B * (w + c) + mu_C * (s+d))./(O+(mu_A + mu_B + mu_C) * ones(size(O)));
 
    
    % ����Լ����Ӧ�������������ճ���(B,C,D)�ĸ��¹�ʽ
    b = b-(x-z); % ����b��Լ��z��
    c = c-(x-w); % ����c��Լ��w��
    d = d-(x-s); % ����d��Լ��s��

    mu_A = 1.01 * mu_A;
    mu_B = 1.01 * mu_B;
    mu_C = 1.01 * mu_C;

    % ����ֹͣ����
    if j>1
        if(All_PSNR(j)-All_PSNR(j-1)<0)
               fprintf('���Ž����iter number = %d, PSNR = %f, SSIM = %f\n',(j-1), All_PSNR(j-1),All_SSIM(j-1));
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

