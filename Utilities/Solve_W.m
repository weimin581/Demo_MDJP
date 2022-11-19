function W = Solve_W(E, P)
% solve the following problem:
% 1/2*||W-E||^2_2+tau/mu_B*||P*(卷积)W||^p_p
% 由ADMM:1/2*||W-E||^2_2+tau/mu_B*||d||^p_p+alpha/2||P*W-d-h||^2_2
% 转化为：1/2*||W-E||^2_2+alpha/2||P*W-d-h||^2_2 和 ita/2*||d-d0||^2_2+1/2||d||^p_p 求解
% 其中 d0=P*W-h

p        =  0.4;  
alpha    =  1;            %拉格朗日系数    
ita      =  0.02;         %ita = alpha*mu_B/tau;默认值0.02;
rho      =  1.05;         %rho = 2*sqrt(2); 
itermax  =  50;
all_psnr =  zeros(1,itermax);
K        =  size(P,1);
[m,n]    =  size(E);
% 初始化
W  = E;
h  = zeros(size(W));
FP = zeros(m,n,K,'single');
for i=1:K
    FP(:,:,i) = psf2otf(reshape(P(i,:),[3,3]),[m,n]);
end
FW = repmat(fft2(W),[1,1,K]);
h  = repmat(h,[1,1,K]);  %拉格朗日乘子
Fh = fft2(h);
Fd = FP.*FW;
FE = fft2(E);
SumFPTP = sum(FP.*conj(FP),3);

for iter = 1:itermax
    FW    =  (FE+alpha*sum(conj(FP).*(Fd+Fh),3))./(ones(size(E))+alpha*SumFPTP);
    W     =  ifft2(FW);

    d0    =  real(ifft2(FP.*repmat(FW,[1,1,K])))-h; 
    d     =  GST( d0, 1/ita, p);
    Fd    =  fft2(d);
    
    h     =  h - (d0 - d);
    Fh    =  fft2(h);
    
    ita   =  rho*ita;
    alpha =  alpha*rho;
    
    all_psnr(iter)  =  csnr(W,E,0,0);
    if iter>1
        if(all_psnr(iter)-all_psnr(iter-1)<0)
            break;
        end
    end
%     figure(3);imshow(uint8(W))
end

