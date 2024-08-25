clc;
clear;
close all;
warning off

addpath ./ClusteringMeasure
addpath ./nonconvex_funs
addpath ./tools
path = './data/';
addpath(genpath('./data/'));

Dataname = 'NGs'; 
percentDel = 0.1;  %  missing rate
Datafold = [Dataname,'_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datafold);
for i=1:length(X)
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);  %normalized
end

%%  parameters 
param.alpha = 0.001;
param.beta = 1; 
param.lambda = 1;
param.p = 0.9;
param.k =  2;
f  = 1 ;
param.iter = 50;
cls_num = numel(unique(gt));

 %%    
        ind_folds = folds{f};      
        [G,LOSS] = CGCTD(X,ind_folds,gt, param); 

%%
        [Clus] = SpectralClustering(G, cls_num);  
        [ACC,NMI,PUR] = ClusteringMeasure(gt,Clus) ;%ACC NMI Purity
         result =   [ACC,NMI,PUR] 





