function [ corrupt_image, x_inpaint_rgb_all , best_psnr , best_ssim]=MDJP_main(ori_gname,MaskType,patch,p_miss,delta,mu_A,mu_B,rou,p)   
       
        % read target img
        x_rgb = imread(ori_gname); 
        % rgb_to_yuv
        x_yuv = rgb2ycbcr(x_rgb);
        
        [m , n , c] = size(x_yuv);
        
        x     = double(x_yuv(:,:,1)); 
        % ground truth: x_org
        x_org = x;
        
        % result
        x_inpaint_re_all           =    zeros(size(x_yuv));
        x_inpaint_re_all(:,:,2)    =    x_yuv(:,:,2); 
        x_inpaint_re_all(:,:,3)    =    x_yuv(:,:,3); 
        
        ratio                  =    p_miss; 

        if MaskType == 1       %random mask;
            rand('seed',0);
            O = double(rand(size(x)) > (1-ratio));
        elseif MaskType == 2   %line mask
            O = imread('line_mask.png');
            O = imresize(O,[m,n]);
            O = double(O>128);  
            O = O(:,:,1); 
        elseif MaskType == 3   %grid mask
            O = imread('grid_mask.png');
             O = imresize(O,[m,n]);
            O = double(O>128); 
            O = O(:,:,1); 
        elseif MaskType == 4 
            O = imread('demomask.png');
             O = imresize(O,[m,n]);
            O = double(O>128); 
            O = O(:,:,1); 
        end
        
        % add noise
        randn('seed',0);
        x = x + delta * randn(size(x));
        y = double(x).* O;       % Observed Image

        para = [];
        
        if ~isfield(para,'mu_A')
            para.mu_A = mu_A;
        end
        
        if ~isfield(para,'mu_B')
            para.mu_B = mu_B;
        end
        
        if ~isfield(para,'org')
            para.org = x_org;
        end  
        
        if ~isfield(para,'IterNums')
            para.IterNums = 40000;
        end 
        
        if ~isfield(para,'Initial')
            para.Initial = Inter_Initial(y,~O);
        end
        
        if ~isfield(para,'patch')
            para.patch = patch; 
        end
        
        if ~isfield(para,'step')
            para.step = 8; 
        end       
        
        if ~isfield(para,'Similar_patch')
            para.Similar_patch = 60; 
        end
         
        if ~isfield(para,'Region')
            para.Region = 25;
        end        
        
        if ~isfield(para,'sigma')
            para.sigma = sqrt(2);
        end 
        
        if ~isfield(para,'e')
            para.e = 0.3;
        end         
                
     fprintf(ori_gname);
     fprintf('\n');

     fprintf('Initial PSNR = %f, SSIM = %f\n',csnr(y,x_org,0,0),ssim(y,x_org));
     
     [reconstructed_image_all ,  best_psnr , best_ssim]  = MDJP_Solved_by_AM_ADMM(y , O , para , p , delta , rou); 

     % corrupt_image
     corrupt_image =  zeros(size(x_yuv));
     corrupt_image(:,:,1) = y;
     corrupt_image(:,:,2) = x_yuv(:,:,2);
     corrupt_image(:,:,3) = x_yuv(:,:,3);
     corrupt_image = ycbcr2rgb(uint8(corrupt_image));
     
     % res£ºx_inpaint_rgb_all
     x_inpaint_re_all(:,:,1) = reconstructed_image_all;
     x_inpaint_rgb_all = ycbcr2rgb(uint8(x_inpaint_re_all));
     fprintf('................................................\n');       
       
end

