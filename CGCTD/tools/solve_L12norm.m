function [E_new]= solve_L12norm(E, lembda)
%%��׼�ͣ�lembda||E_new||_{1,2}^2+||E_new-E||_F^2
%%���룺lembda��E
%%�����E_new
%%%%%%% ������� E_j  %%%%%%   
[m,n] = size(E);
E_new = zeros(m,n);
%%%%%%%   �������L12���� %%%%%%%
for k=1:n
    e = E(:,k);
    [tao, mu] = search_tao_mu(e, lembda);
    e_temp = abs(e) - lembda*tao/(1.0 + lembda*tao)*mu;
    e_temp(find(e_temp<0)) = 0;
    E_new(:,k) = sign(e).*e_temp;
end

function [tao,mu] = search_tao_mu(e, lembda)
%
d = length(e);
e_abs = abs(e);
[temp,S] = sort(e_abs,'descend');
% Initialize
tao = d;
mu = 1.0/d*sum(e_abs);
while (tao>1) && (e_abs(S(tao)) - lembda*tao/(1+lembda*tao)*mu <0)
    mu = tao/(tao-1)*mu - 1.0/(tao-1)*e_abs(S(tao));
    tao = tao-1;
end



