function [G,LOSS] = ET(X, gt, param)


num_view = length(X);    %视图数
N = numel(gt);       %样本数

alpha= param.alpha;
beta   =  param.beta;
lambda  = param.lambda;
p=param.p;
Obj=[];
weight_vector = ones(1,num_view)'; 
for iv = 1:num_view
    H{iv} = zeros(N);  
    E{iv} = zeros(N); 
    Z{iv} = zeros(N);   
    Y{iv} = zeros(N); 
    ed{iv} = L2_distance_1(X{iv}, X{iv});  
end

sX = [N, N ,num_view];
MAX_iter = 55;

rho = 1e-3;
dt = 1.3;

for iter = 1:MAX_iter
% for iter = 1:param.iter 

%% update S^v
        for i = 1:num_view                             
        temp_A=zeros(N);
        for j = 1:N
            ad = (rho*(H{i}(j,:)+E{i}(j,:))-ed{i}(j,:)-Y{i}(j,:))/(2*alpha+rho);
            temp_A(j,:) = EProjSimplex_new(ad);
        end
        S{i} = temp_A;      
        end
		
%% update H
        for v =1:num_view
        QQ{v}=(S{v} -E{v} + Y{v}/rho);
        end
        Q_tensor = cat(3,QQ{:,:});
        Qg = Q_tensor(:);
%         [myj, ~] = wshrinkObj_weight_lp(Qg, weight_vector./rho,sX, 0,3,p);
        [myj, ~] = wshrinkObj_weight_lp(Qg, (beta*weight_vector)./rho,sX, 0,3,p);
        H_tensor = reshape(myj, sX);
        for k=1:num_view
        H{k} = H_tensor(:,:,k);
        end
		
%% update E 
        for i=1:num_view
        E{i} = prox_l1(S{i}-H{i}+Y{i}/rho, lambda/rho);
        end
%%		
    RR1 = [];
    for i=1:num_view
        res1 = S{i}-H{i}-E{i};       
        Y{i} = Y{i} + rho * res1;     
        RR1 =[RR1, norm(res1,'inf')];       
    end
    rho = min(1e6, dt*rho);
    thrsh = 1e-5;
%     if(norm(RR1, inf)<thrsh )
%        break;
%     end
    loss1(iter) = norm(RR1, inf); 
    RR1 = []; 
  
end

KK=0;
 for i=1:num_view
    KK = KK + (abs(H{i})+(abs(H{i}))');
%     KK = KK + (H{i}+(H{i})');
 end
G = KK/2/num_view;
LOSS=loss1;


end

