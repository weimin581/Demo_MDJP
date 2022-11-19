function X = DWT2DCT(x,Dct_matrix,W_matrix)

[M,N,K]=size(x);
y = zeros(K,M*N);
X = zeros(K,M*N);
temp = zeros(M,N);

for k=1:K
     temp = W_matrix*x(:,:,k)*W_matrix';
     y(k,:)=temp(:)';
end

X = Dct_matrix*y;


