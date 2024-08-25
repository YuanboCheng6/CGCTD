function [G,LOSS] = CGCTD(X, ind_folds,gt, param)


num_view = length(X);    %视图数
N = numel(gt);       %样本数
cls_num = numel(unique(gt));
A=[];

alpha = param.alpha;
beta  =  param.beta;
lambda  = param.lambda;
p = param.p;
k = param.k;
Obj=[];
weight_vector = ones(1,num_view)'; 
for iv = 1:num_view
    H{iv} = zeros(N);  
    E{iv} = zeros(N); 
    Z{iv} = zeros(N);   
    Y1{iv} = zeros(N); 
end

sX = [N, N ,num_view];
max_iter = param.iter;

rho = 1e-3;
dt = 1.3;

%%  构造不完备数据 Y 和索引矩阵 G

    Xc = X;
    for i=1:length(X)
        Xci = Xc{i};
        indi = ind_folds(:,i);
        pos = find(indi==0);
        Xci(:,pos)=[];
        Y{i} = Xci;
    end
    G = cell(1, num_view); 
	for i=1:num_view
    pos = find(ind_folds(:,i)==1);
    T= zeros(N, length(pos));
    for j=1:length(pos)
        T(pos(j),j) = 1;
    end
    G{i} = T';
	end



%%  初始化  Z^v
for iv = 1:length(X)
    X1 = Y{iv};
    % ---------- knn -----  Binary ----- %
    options = [];
    options.NeighborMode = 'KNN';
    options.k = k;
    options.WeightMode = 'Binary';      
    Z1 = full(constructW(X1',options));
    Z1 = Z1- diag(diag(Z1));
    Z{iv} = max(Z1,Z1');
    Z{iv} = G{iv}'*Z{iv}*G{iv};
    clear Z1 
	
end

%%    update W^v
for iv = 1:num_view
    options.WeightMode = 'Binary';  % Binary
    Z1 = full(constructW(Y{iv}',options));
    Z1 = Z1+eye(size(Z1));
    Z1 = Z1*Z1;
    Z1 = Z1./max(Z1(:));
    W{iv} = G{iv}'*Z1*G{iv};    % 置信W^v 矩阵
    clear Z1
end

%%
for iv = 1:num_view
ed{iv} = L2_distance_1(Y{iv}, Y{iv});  
end
%%
for iter = 1:max_iter

%%  update S^v

    for iv = 1:num_view
        ind_0 = find(ind_folds(:,iv)==0);
        linshi_B = Z{iv};
        linshi_B(ind_0,:) = [];
        linshi_B(:,ind_0) = [];                
        linshi_S = (2*alpha*linshi_B-ed{iv})/(2*alpha);
        linshi_S2 = zeros(size(linshi_S));
        for in = 1:size(linshi_S,2)
            idx = [1:size(linshi_S,2)];
            idx(in) = [];    
            linshi_S2(idx,in) = EProjSimplex_new(linshi_S(idx,in)); 
        end
        S{iv} = linshi_S2;        
    end  


%%  update  Z^v


    for iv = 1:num_view
        linshi_GSG = G{iv}'*S{iv}*G{iv};      
        linshi_Z = (2*alpha*linshi_GSG.*(W{iv}.*W{iv})+ rho*(H{iv}+E{iv})- Y1{iv})./(2*alpha*(W{iv}.*W{iv})+ rho);
        linshi_Z2 = zeros(size(linshi_Z));
        for in = 1:size(linshi_Z,2)
            idx = [1:size(linshi_Z,2)];
            idx(in) = [];
            linshi_Z2(idx,in) = EProjSimplex_new(linshi_Z(idx,in));      
        end
        Z{iv} = linshi_Z2;
    end  
	
%% update H
        for v =1:num_view
        QQ{v}=(Z{v} - E{v} + Y1{v}/rho);
        end
        Q_tensor = cat(3,QQ{:,:});
        Qg = Q_tensor(:);
        [myj, ~] = wshrinkObj_weight_lp(Qg, (beta*weight_vector)./rho, sX, 0,3,p);
        H_tensor = reshape(myj, sX);
        for k=1:num_view
        H{k} = H_tensor(:,:,k);
        end

%% update E 
        for i=1:num_view
        E{i} = prox_l1(Z{i}-H{i}+Y1{i}/rho, lambda/rho);
        end
%%		
    RR1 = [];
    for i=1:num_view
        res1 = Z{i}-H{i}-E{i};
        Y1{i} = Y1{i} + rho * res1;     
        RR1 =[RR1, norm(res1,'inf')];         
    end

    rho = min(1e6, dt*rho);      
    loss1(iter) = norm(RR1, inf); 
end
LOSS=loss1;
KK=0;

 for i=1:num_view
    KK = KK + (abs(H{i})+(abs(H{i}))');
 end
G = KK/2/num_view;




end