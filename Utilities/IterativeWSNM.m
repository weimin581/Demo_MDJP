function [ SigmaX, svp ] = IterativeWSNM( SigmaY, C, p )
% weighted schatten p-norm minimization

p1    =   p; %0.7   
Temp  =   SigmaY;%5*1
s     =   SigmaY;%5*1
s1    =   zeros(size(s));%5*1

for i=1:3

   W_Vec    =   C;
   [s1, svp]=   solve_Lp_w(s, W_Vec, p1);
   Temp     =   s1;
end
SigmaX = s1;

end

