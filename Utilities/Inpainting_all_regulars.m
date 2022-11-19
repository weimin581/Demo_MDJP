function [reconstructed_image_all , All_PSNR]= Inpainting_all_regulars(y,O,para,p)

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

    All_PSNR(j) = csnr(x,x_org,0,0);
    All_SSIM(j) = ssim(x,x_org);
    
    iter(j) = j;
    
    % ��һ��Լ���� X = Z(WSNM)
    z = Solve_Z_GST(x-b,para,p); 

    % �ڶ���Լ���� X~ = W(ASR)
    w = Solve_W2_new(x-c,P,mu_B);  

    % ������Լ����X = S (NLSM)
    s = Solve_NLSM(x-d,Opts);   %�¼�һ�����NLSM�ķ���

    % �ܵļ��㹫ʽ
    x = (y + mu_A * (z + b)+ mu_B * (w + c) + mu_C * (s+d))./(O+(mu_A + mu_B + mu_C) * ones(size(O)));
    
    
    % show temp results
    % imshow([z/255, w/255, s/255, x/255],[0,1]);
    
    % ����Լ����Ӧ�������������ճ���(B,C,D)�ĸ��¹�ʽ
    b = b-(x-z); % ����b��Լ��z��
    c = c-(x-w); % ����c��Լ��w��
    d = d-(x-s);% ����d��Լ��s��

    mu_A = 1.01 * mu_A;
    mu_B = 1.01 * mu_B;
    mu_C = 1.01 * mu_C;

    % ����ֹͣ����
    if j>1
        if(All_PSNR(j)-All_PSNR(j-1)<0)
               fprintf('���ս���� PSNR = %f, SSIM = %f\n',All_PSNR(j-1),All_SSIM(j-1));
            break;
        end
    end

    %fprintf('iter number = %d, PSNR = %f, FSIM = %f\n',j,csnr(x,x_org,0,0),FeatureSIM(x_org,x));
    fprintf('x: iter number = %d, PSNR = %f, SSIM = %f\n',j,csnr(x,x_org,0,0),ssim(x,x_org));
    
end
toc

x(x>255) = 255;
x(x<0) = 0;   

% ע�Ⱑ����������ʱ�򣬲�Ҫ�ǣ�x 
reconstructed_image_all = x;

end

