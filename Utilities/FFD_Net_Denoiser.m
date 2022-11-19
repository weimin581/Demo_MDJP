function  [im_denoised]     =  FFD_Net_Denoiser (input, imageNoiseSigma)

randn ('seed',0);

inputNoiseSigma   =   imageNoiseSigma;

format compact;

global sigmas;

% 注意啊：因为只在YUV的第一个通道进行图像修复，所以在此调用的是FFDNet的单通道版本！！！
load(fullfile('models','FFDNet_gray.mat'));

net = vl_simplenn_tidy(net);

% 此时的输入信号：input
input = double(input)/255;
   
  
sigmas = inputNoiseSigma/255; 
    
%Option 1: 使用自己写的函数，不用matconvet库！
 res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow
 
%Option 2: 这个问题先放放，使用Matconvet计算可以提高速度！！！
%  res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default  
 
output = res(end).x;
    
im_denoised  =  double(output*255);

end

