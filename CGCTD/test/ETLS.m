function [G,loss1] = ETLS(X, gt, param)

% loss1=0;
% loss2=0;
% loss3=0;
% obj=0;
num_view = length(X);    %视图数
N = numel(gt);       %样本数
	alpha= param.alpha;
	beta = param.beta;
	lambda  = param.lambda;
	gamma   =  param.gamma;
	p = param.p;	   
	weight_vector = ones(1,num_view)'; 
    for iv = 1:num_view
		A{iv} = zeros(N);  
		F1{iv} = zeros(N); 
		ed{iv} = L2_distance_1(X{iv}, X{iv});   
    end
    B = A;
    E = A;
	H = A;
    S = A;
    F2 = F1;
    F3 = F1;
    sX = [N, N ,num_view];
    MAX_iter = 100;
%     MAX_iter = param.iter;

    rho = 1e-3;
	dt = 1.3;
for iter = 1:MAX_iter
%%   update S^v 

    for i=1:num_view
        temp_S=zeros(N); 
        for j = 1:N
            ad = (rho*(H{i}(j,:)+E{i}(j,:))-ed{i}(j,:)-F1{i}(j,:))/(2*alpha+rho);
            temp_S(j,:) = EProjSimplex_new(ad);
        end
        S{i} = temp_S;       	  	 	  
    end 
%%   update A^v
    for v =1:num_view
        temp{v}=(H{v} + F2{v}/rho);
    end
    temp_tensor = cat(3,temp{:,:});
    Qg = temp_tensor(:);
    [myj, ~] = wshrinkObj_weight_lp(Qg, (beta*weight_vector)./rho,sX, 0,3,p);
    A_tensor = reshape(myj, sX);
    for k=1:num_view
        A{k} = A_tensor(:,:,k);
    end	
%%   update B^v
    H_tensor = cat(3, H{:,:});
    F3_tensor = cat(3, F3{:,:});
    B_tensor  = cat(3, B{:,:});
    tempBF4 = H_tensor +  F3_tensor/rho;
    for i=1:N	
%         B_tensor(i,:,:)  = prox_l21(reshape(tempBF4(i,:,:), [num_view, N])', gamma/rho);
%         B_tensor(i,:,:) = prox_l1(reshape(tempBF4(i,:,:), [num_view, N])', gamma/rho);
        B_tensor(i,:,:)  = solve_L12norm(reshape(tempBF4(i,:,:), [num_view, N])', gamma/rho);
    end
    for i=1:num_view
        B{i} = B_tensor(:,:,i);		
    end

%%   update E^v

    for i=1:num_view
        E{i} = prox_l1(S{i}-H{i}+F1{i}/rho, lambda/rho);
%         E{i} = solve_L12norm(S{i}-H{i}+F1{i}/rho, lambda/rho);
    end

%%   update H^v
    for i=1:num_view
        H{i} = (S{i}+ A{i}+ B{i}- E{i}+ F1{i}/rho- F2{i}/rho- F3{i}/rho )/3;
    end 
%%   update  F1 F2 F3
    RR1 = [];RR2=[]; RR3=[];    obj3 = 0;
    obj2 = 0;
    for i=1:num_view
        res1 = S{i} - H{i} - E{i};
        res2 = H{i} - A{i};
        res3 = H{i} - B{i};
        F1{i} = F1{i} + rho*res1;
        F2{i} = F2{i} + rho*res2;
        F3{i} = F3{i} + rho*res3;
        RR1 =[RR1,  norm(res1,'inf')];
%         RR2 =[RR2,  norm(res2,'inf')];
%         RR3 =[RR3,  norm(res3,'inf')];
%         L{i} = diag(sum(S{i})) - S{i};
%         obj2 = obj2 + trace(X{i}*L{i}*X{i}');
%         obj3 = obj3 + norm(S{i},'fro')^2;   
    end
    loss1(iter) = norm(RR1, inf);     
%     loss2(iter) = norm(RR2, inf);
%     loss3(iter) = norm(RR3, inf);
%     obj(iter) = obj2 + alpha*obj3;
    RR1 = [];RR2=[]; RR3=[];
    rho = min(1e6, dt*rho);	
end


KK=0;
 for i=1:num_view
    KK = KK + (abs(A{i})+(abs(A{i}))');
    %KK = KK + A{i}+(A{i})';
 end
G = KK/2/num_view; 

end