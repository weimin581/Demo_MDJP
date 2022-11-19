
function [Imout] = Solve_GSR(InputImage, Options)

% 来自JPG-SR中的设定参数
Options.mu = 0.2 ; % Parameter for PSR
Options.Thr = 8;
% Options.true = true;
Options.Region = 25;
Options.patch = 8;
Options.Sim = 60;

Options.step = 8; %要变步长_2(提升GSR的计算速度)

%这个就是SNSS中随着迭代次数变化的噪声参数
Options.sigma = sqrt(2);

Options.c1 = 0.2;

[H, W]                  =      size(InputImage);

Region                  =      Options.Region;

patch                   =      Options.patch;

Patchsize               =      patch*patch;

Sim                     =      Options.Sim;

step                    =      Options.step;

N                       =      H-patch+1;

M                       =      W-patch+1;

L                       =      N*M;

row                     =       [1:step:N];

row                     =       [row row(end)+1:N];

col                     =       [1:step:M];

col                     =       [col col(end)+1:M];

Groupset                =       zeros(Patchsize, L, 'single');

cnt                     =         0;

for      i  = 1:patch
    
    for      j  = 1:patch
        
             cnt    =  cnt+1;
             Patch  =  InputImage(i:H-patch+i,j:W-patch+j);
             Patch  =  Patch(:);
             Groupset(cnt,:) =  Patch';
    end
    
end

GroupsetT               =           Groupset';

I                       =           (1:L);

I                       =           reshape(I, N, M);

NN                      =           length(row);

MM                      =           length(col);

Imgtemp                 =            zeros(H, W);

Imgweight               =            zeros(H, W);

Array_Patch             =            zeros(patch, patch, Sim);


for  i  =  1 : NN
    
    for  j  =  1 : MM
        
           currow              =              row(i);
           
            curcol             =              col(j);
            
             off               =              (curcol-1)*N + currow;
      
            Patchindx          =               Similar_Search(GroupsetT, currow, curcol, off, Sim, Region, I);
        
            curArray           =               Groupset(:, Patchindx);
        
             M_temp            =               repmat(mean(curArray,2),1,Sim);
        
             curArray          =               curArray-M_temp;
        
            U_i                =               getsvd(curArray); % generate PCA basis
        
            A0                 =               U_i'*curArray;
        
            s0                 =               mean (A0.^2,2);

            s0                 =               max  (0, s0-Options.sigma^2);    
            
            lambda             =               repmat ( 2*sqrt(2)*Options.c1*Options.sigma^2./(sqrt(s0)+eps),[1, size(A0,2)]); %Generate the weight Eq.(19)
        
            % 参数随着迭代次数变化
            tau               =               lambda*step*Sim/(Options.mu); 
             
           
            Thre               =               sqrt(2*tau);
        
           Alpha               =               A0.*(abs(A0)>Thre);  % L0_norm (soft operation)

          curArray             =               U_i*Alpha   +   M_temp;
            
        
        for k = 1:Sim
            
            Array_Patch(:,:,k) = reshape(curArray(:,k),patch,patch);
            
        end
        
        for k = 1:length(Patchindx)
            
            RowIndx  =  ComputeRowNo((Patchindx(k)), N);
            
            ColIndx  =  ComputeColNo((Patchindx(k)), N);
            
            Imgtemp(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1)    =   Imgtemp(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1) + Array_Patch(:,:,k)';
         
            Imgweight(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1)  =   Imgweight(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1) + 1;
        
        end
        
    end
    
end

Imout = Imgtemp./(Imgweight+eps);

return;



