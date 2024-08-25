clc;
clear;
close all;

addpath ./ClusteringMeasure
addpath ./nonconvex_funs
path = './data/';

% load  ./data/BBC4view
% name = 'BBC4view';
% load  ./data/3sources
% name = '3sources';
% load  ./data/MSRC   %1 10 1 1 0.3
% name = 'MSRC';
% load  ./data/WebKB
% name = 'WebKB';
% load  ./data/20newsgroups  % 10  10  1  1  0.3
% name = '20newsgroups';
% load  ./data/BBC_2view   % 10 100 1  1 0.3
% name = 'BBC_2view';
% load  ./data/ORL
% name = 'ORL';
load  ./data/yale_newdouble  %0.01 0.01 0.1 100 0.3
name = 'yale_newdouble';

A_max  = 0;
A_flag = 0;


    for i=1:length(X)
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);  %normalized
    end

% for i = [0.001,0.01,0.1,1,10,100,1000]
% for i=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
%  for i =10: 100
param.alpha = 0.001;
param.beta = 0.01;
param.lambda = 0.1;
param.gamma = 1;
param.p= 0.4;

% param.iter = 10 ; 

cls_num = numel(unique(Y));

perf = []; 
gt = double(Y);
   for t = 1 : 1
       
       
     [G,loss1] = ETLS(X, gt, param) ;
  
     [Clus] = SpectralClustering(G, cls_num);  
        [ACC,NMI,PUR] = ClusteringMeasure(gt,Clus); %ACC NMI Purity
        [Fscore,Precision,R] = compute_f(gt,Clus);
        [AR,~,~,~]=RandIndex(gt,Clus);
        result = [ACC NMI PUR  Fscore  Precision R AR];
        perf  = [perf; result*100];
%         fprintf(" ACC,NMI,PUR,FS,PR,RE: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n",  result(1),result(2),result(3),result(4),result(5),result(6)); 
A1(t)=ACC;
A2(t)=NMI;
A3(t)=PUR;
A4(t)=Fscore;
A5(t)=Precision;
A6(t)=R;
   end

ACC_mean = roundn(mean(A1),-4); ACC_std = roundn(std(A1),-4);
NMI_mean = roundn(mean(A2),-4); NMI_std = roundn(std(A2),-4);
PUR_mean = roundn(mean(A3),-4); PUR_std = roundn(std(A3),-4);
FS_mean = roundn(mean(A4),-4); FS_std = roundn(std(A4),-4);
PR_mean = roundn(mean(A5),-4); PR_std = roundn(std(A5),-4);
RE_mean =roundn(mean(A6),-4); RE_std = roundn(std(A6),-4);
Mean = [ACC_mean  NMI_mean PUR_mean  FS_mean  PR_mean  RE_mean ]
Std = [ACC_std  NMI_std  PUR_std  FS_std PR_std RE_std]
% 

if (ACC_mean>A_max)
  A_max=ACC_mean;
  A_flag=i;
end
%   end


