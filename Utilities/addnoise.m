function [nimg,ind]=addnoise(img,r,type)
%Add noise to img.
%[nimg,ind]=addnoise(img,r,type)
%r is the percentage of the impulse noise.
%type  'sp' salt and pepper
%      'rd' random impulse noise.

%rand('state',sum(100*clock));
[n,m]=size(img);
nimg=img;
rand('seed',19);
ran=rand(n,m);
ind=zeros(n,m);

index=find(ran<=r/2);
ind(index)=1;
if type=='sp'
    nimg(index)=0;
elseif type=='rd'
    nimg(index)=255*rand(size(index));
end
index=find(ran>=(1-r/2));
ind(index)=1;
if type=='sp'
    nimg(index)=255;
elseif type=='rd'
    nimg(index)=255*rand(size(index));
end
%fig(nimg,'Noised Image')
%disp(['The PSNR of the noised image is ' num2str(psnr(img,nimg))]);