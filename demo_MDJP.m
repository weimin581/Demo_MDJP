% This is the demo for our MDJP, a mixed degradation image completion
% framework; run this demo, you can obtain the result in our paper of
% Fig.1. standard test image Butterfly mixed degradation by AWGN (30) and irregular scratches.  
% MDJP is submitted to TCYB.

clc;
clear;
close all;
addpath('./Test_Images/');
addpath('./Utilities/');
addpath('./three_datasets/CBSD68/');
% addpath('./three_datasets/Set12/');
addpath('./three_datasets/Kodak24/');


for i = 1 : 1 : 1
    
    for ImgNo = i
        switch ImgNo
            
            case 1
                filename = 'Butterfly';
            case 2
                filename = 'Girl';

        end
        orgname = [filename '.png'];
        
        % we conduct 6 mixed degradation modes in our simulation:
        % Noisy pixep-wise missing: delta 30 with 50% missing,  delta 40 with 40% missing,  delta 50 with 30% missing;
        % Noisy mask covering: delta 30 with line,  delta 40 with grid,  delta 50 with test;
        delta     =  30;       % % noise level : 30 / 40 / 50 
        p_miss    =  0.5;      % missing ratio 0.5£¨50%missing£©/ 0.6£¨40%missing£©/ 0.7£¨30%missing£© 
        
        MaskType  =  4;        % irregular scratches in Fig.1 
        
        mu_A = 0.5;
        mu_B = 0.5;
        rou  = 1.01;
        p = 0.95;   

        fprintf('ImgNo = %f\n',ImgNo);
        
        [corrupt_image, im_out_all , best_psnr , best_ssim]  =  MDJP_main(orgname, MaskType, 8, p_miss,delta,mu_A,mu_B,rou,p);

    end
       
       imshow([corrupt_image , im_out_all],[0,1]);


end


