function X = IDWT2DCT(x,Dct_matrix,IW_matrix)
[K,MN]=size(x);
block_size = 8;
X = zeros(block_size,block_size,K);
temp = zeros(block_size,block_size);

y = (Dct_matrix'*x);

for k = 1:K
    temp = reshape(y(k,:),block_size,block_size);
    X(:,:,k) = IW_matrix*temp*IW_matrix';    
end
