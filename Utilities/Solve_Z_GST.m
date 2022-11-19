function [Imout] = Solve_Z_GST(InputImage, para, p)

[h, w]            =              size(InputImage);

region            =              para.Region;

patch             =              para.patch;

patchsize         =              patch*patch;

sim_patch         =              para.Similar_patch;

step              =              para.step;

n                 =              h-patch+1;

m                 =              w-patch+1;

L                 =              n*m;

row               =              [1:step:n];

row               =              [row row(end)+1:n];

col               =              [1:step:m];

col               =              [col col(end)+1:m];

groupset          =              zeros(patchsize, L, 'single');

cnt               =               0;

for i  = 1:patch
    for j  = 1:patch
        cnt    =  cnt+1;
        Patch  =  InputImage(i:h-patch+i,j:w-patch+j);
        Patch  =  Patch(:);
        groupset(cnt,:) =  Patch';
    end
end

GroupsetT        =               groupset';

I                =               (1:L);

I                =                reshape(I, n, m);

NN               =                length(row);

MM               =                length(col);

Imgtemp          =                zeros(h, w);

Imgweight        =                zeros(h, w);

Array_Patch      =                zeros(patch, patch, sim_patch);


for  i  =  1 : NN
    for  j  =  1 : MM
        
        currow          =           row(i);
        
        curcol          =           col(j);
        
        off             =           (curcol-1)*n + currow;
        
        Patchindx       =           Similar_Search(GroupsetT, currow, curcol, off, sim_patch, region, I);
        
        curArray        =           groupset(:, Patchindx);
        
        M_temp          =           repmat(mean(curArray,2),1,sim_patch);
       
        curArray        =           curArray-M_temp;
        
     [GR_S, GR_V, GR_D] =           svd(double(full(curArray)),'econ');  %svd(CurArray);

        cc             =            sqrt( mean(GR_V.^2, 2) );
        
        lambda         =            2*sqrt(2)*para.sigma^2./(cc + para.e);
        
        tau            =            lambda*step*sim_patch/para.mu_A;
        
        Thre           =            1./(diag(GR_V)+0.1);
        
        GR_Z           =            IterativeWSNM(GR_V,diag(tau.*Thre),p);  % Eq. (16) P-norm ÖÐµÄp = 0.95
        
        curArray       =            GR_S*GR_Z*GR_D'+M_temp;
        
        for k = 1:sim_patch
            Array_Patch(:,:,k) = reshape(curArray(:,k),patch,patch);
        end
        
        for k = 1:length(Patchindx)
            RowIndx  =  ComputeRowNo((Patchindx(k)), n);
            ColIndx  =  ComputeColNo((Patchindx(k)), n);
            Imgtemp(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1)    =   Imgtemp(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1) + Array_Patch(:,:,k)';
            Imgweight(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1)  =   Imgweight(RowIndx:RowIndx+patch-1, ColIndx:ColIndx+patch-1) + 1;
        end
        
    end
end

Imout = Imgtemp./(Imgweight+eps);

return;



