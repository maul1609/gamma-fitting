function [n0nlin,munlin,lamnlin]=myfiting_test(d_size,now_data_DP)

[r,c]=size(now_data_DP);
dedge=[2,d_size];

beta0=[log10(1e12),2,log10(1e3)]

n0nlin=zeros(r,1);
munlin=zeros(r,1);
lamnlin=zeros(r,1);
for i=1:r
    beta = lsqcurvefit(@(beta,xdata) fun(beta,xdata),beta0,...
        d_size./1e6,now_data_DP(i,:).*1e12./diff(dedge));
    n0nlin(i)=beta(1);
    munlin(i)=beta(2);
    lamnlin(i)=beta(3);
end

function F=fun(beta,xdata)

F=abs(10.^beta(1)).*xdata.^abs(beta(2)).*exp(-abs(10.^beta(3)).*xdata);
