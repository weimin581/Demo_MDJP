function  [im_denoised]     =  FFD_Net_Denoiser (input, imageNoiseSigma)

randn ('seed',0);

inputNoiseSigma   =   imageNoiseSigma;

format compact;

global sigmas;

% ע�Ⱑ����Ϊֻ��YUV�ĵ�һ��ͨ������ͼ���޸��������ڴ˵��õ���FFDNet�ĵ�ͨ���汾������
load(fullfile('models','FFDNet_gray.mat'));

net = vl_simplenn_tidy(net);

% ��ʱ�������źţ�input
input = double(input)/255;
   
  
sigmas = inputNoiseSigma/255; 
    
%Option 1: ʹ���Լ�д�ĺ���������matconvet�⣡
 res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow
 
%Option 2: ��������ȷŷţ�ʹ��Matconvet�����������ٶȣ�����
%  res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default  
 
output = res(end).x;
    
im_denoised  =  double(output*255);

end

