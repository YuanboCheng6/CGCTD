folds = cell(1,10);
for i=1:10   % 
    for j=1:3   % 视图数
a  = ones(20,1);  % 可用样本数
b  = zeros(20,1);  % 缺失样本数
c = [a;b];
rowrank = randperm(size(c, 1)); % size获得a的行数，randperm打乱各行的顺序
c = c(rowrank,:);              % 按照rowrank重新排列各行，注意rowrank的位置
folds{1,i}(:,j)=c;
    end
end