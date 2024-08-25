% function [G, A,S,loss1,loss2,loss3,obj] = ETLSRR(X, gt, param)
function [G, A,S,loss1] = ETLSRR(X, gt, param)
num_view = length(X);    %视图数
N = numel(gt);       %样本数

lambda  = param.lambda;
theta   =  param.theta;
mu = param.mu;
alpha= param.alpha;
Obj=[];
weight_vector = ones(1,num_view)'; 
for iv = 1:num_view
    A{iv} = zeros(N);  
    Z{iv} = zeros(N);   
    F2{iv} = zeros(N); 
    ed{iv} = L2_distance_1(X{iv}, X{iv});  
end
B = A;
C = A;
S  = F2;
F3 = F2;
F4 = F3;
sX = [N, N ,num_view];
MAX_iter = 80;

rho = 1e-3;
dt = 1.3;

for iter = 1:MAX_iter
    
     %% Zi
        for i = 1:num_view                             
        temp_A=zeros(N);
        tmp_ed=ed{i}; 
        for j = 1:N
            ad = (rho*(B{i}(j,:)+S{i}(j,:))-tmp_ed(j,:)-F2{i}(j,:))/(2*alpha+rho);
            temp_A(j,:) = EProjSimplex_new(ad);
        end
        Z{i} = temp_A;      
        end
           
    %% A step  
    
%   非凸核范数    
%     B_tensor = cat(3, B{:,:});
%     F3_tensor = cat(3, F3{:,:});
%     Bv = B_tensor(:);
%     F3v = F3_tensor(:);
%     [Av, ~] = wshrinkObj_nc(param.fun, param.p, Bv + 1/rho*F3v, lambda/rho, [N, N, num_view], 0, 3);
%     A_tensor = reshape(Av, [N, N, num_view]);  
%     for i=1:num_view
%         A{i} = A_tensor(:,:,i);
%     end

%     SP范数
%     B_tensor = cat(3, B{:,:});
%     F3_tensor = cat(3, F3{:,:});
%     Bv = B_tensor(:);
%     F3v = F3_tensor(:);   
%     [myj, ~] = wshrinkObj_weight_lp(Bv + 1/rho*F3v, (lambda*weight_vector)./rho,sX, 0,3,param.p);
%     A_tensor = reshape(myj, sX);
%     for k=1:num_view      
%         A{k} = A_tensor(:,:,k);
%     end
%   
%  核范数
    B_tensor = cat(3, B{:,:});
    F3_tensor = cat(3, F3{:,:});
    Bv = B_tensor(:);
    F3v = F3_tensor(:); 
    [Lv, ~] = wshrinkObj(Bv - 1/rho*F3v, lambda/rho, [N, N, num_view], 0, 3);
    A_tensor = reshape(Lv, [N, N, num_view]);
    for i=1:num_view
        A{i} = A_tensor(:,:,i);
    end


    %% S step   噪音
    for i=1:num_view
        S{i} = prox_l1(Z{i}-B{i}+F2{i}/rho, theta/rho);
%         S{i} = solve_L12norm(Z{i}-B{i}+F2{i}/rho, theta/rho);
    end
   
    %% C step
    B_tensor = cat(3, B{:,:});
    F4_tensor = cat(3, F4{:,:});
    C_tensor  = cat(3, C{:,:});
    tempBF4 = B_tensor +  F4_tensor/rho;
    for i=1:N
        C_tensor(i,:,:)  = prox_l21(reshape(tempBF4(i,:,:), [num_view, N])', mu/rho);  %C_tensor(i,:,:) 
%         C_tensor(i,:,:)  = solve_L12norm(reshape(tempBF4(i,:,:), [num_view, N])', mu/rho);
    end
    for i=1:num_view
        C{i} = C_tensor(:,:,i);
    end
  
    %% B  （L）
    for i=1:num_view
        B{i} = (Z{i}- S{i}+ A{i}+ C{i} + F2{i}/rho - F3{i}/rho  - F4{i}/rho)/3;
    end
      
    %% update    

    RR1 = [];RR2=[]; RR3=[];RR4=[];
    obj3 = 0;
    obj2 = 0;
    for i=1:num_view
        res2 = Z{i}- B{i} -S{i};
        res3 = B{i} - A{i};
        res4 = B{i} - C{i};
        F2{i} = F2{i} + rho*res2;
        F3{i} = F3{i} + rho*res3;
        F4{i} = F4{i} + rho*res4;
        RR2 =[RR2,  norm(res2,'inf')];
%         RR3 =[RR3,  norm(res3,'inf')];
%         RR4 =[RR4,  norm(res4,'inf')];
%         L{i} = diag(sum(Z{i})) - Z{i};
%         obj2 = obj2 + trace(X{i}*L{i}*X{i}');
%         obj3 = obj3 + norm(Z{i},'fro')^2;     
    end
     rho = min(1e6, dt*rho); 
     loss1(iter) = norm(RR2, inf);   
%      loss2(iter) = norm(RR3, inf);
%      loss3(iter) = norm(RR4, inf);
%      obj(iter) = obj2 + alpha*obj3;
     RR1 = [];RR2=[]; RR3=[];RR4=[];
%    if( iter>15 && abs(Obj(iter-1)-Obj(iter))< thrsh)
%         break;
%    end
end

KK=0;
 for i=1:num_view
    KK = KK + (abs(A{i})+(abs(A{i}))');
%     KK = KK + (A{i}+(A{i})');
 end
G = KK/2/num_view;

end

 


