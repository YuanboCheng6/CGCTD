folds = cell(1,10);
for i=1:10   % 
    for j=1:3   % ��ͼ��
a  = ones(20,1);  % ����������
b  = zeros(20,1);  % ȱʧ������
c = [a;b];
rowrank = randperm(size(c, 1)); % size���a��������randperm���Ҹ��е�˳��
c = c(rowrank,:);              % ����rowrank�������и��У�ע��rowrank��λ��
folds{1,i}(:,j)=c;
    end
end