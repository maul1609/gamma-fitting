[r,c]=size(now_data_DP);
M0=now_total_Nd.*1e6;
M2=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^2,[r, 1]),2);
M3=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^3,[r, 1]),2);
M6=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^6,[r, 1]),2);

% F=M(2)^3/(M(3)^2*M(0))
% (1-F)*mu^2+(3-6F)*mu+(2-9F)=0

% lam=gamma(mu+4)/gamma(mu+3)*M(2)/M(3)
% n0=M(0)*lam^(mu+1)/gamma(mu+1)

% calculate mu for all data
mu1=zeros(r,1);
mu2=zeros(r,1);
for i=1:r
    % calculate F
    F=M2.^3./(M3.^2.*M0);
    % find the roots
    muroots=roots([(1-F(i)),(3-6.*F(i)),(2-9.*F(i))]);
    mu1(i)=muroots(1);
    mu2(i)=muroots(2);
end

% calculate lambda
lambda1=gamma(mu1+4)./gamma(mu1+3).*M2./M3;
lambda2=gamma(mu2+4)./gamma(mu2+3).*M2./M3;
% calculate n0
n01=M0.*lambda1.^(mu1+1)./gamma(mu1+1);
n02=M0.*lambda2.^(mu2+1)./gamma(mu2+1);


% recalculating M0, M2, M3
M0calc1=n01.*gamma(mu1+1)./lambda1.^(mu1+1);
M0calc2=n02.*gamma(mu2+1)./lambda2.^(mu2+1);
M2calc1=n01.*gamma(mu1+3)./lambda1.^(mu1+3);
M2calc2=n02.*gamma(mu2+3)./lambda2.^(mu2+3);
M3calc1=n01.*gamma(mu1+4)./lambda1.^(mu1+4);
M3calc2=n02.*gamma(mu2+4)./lambda2.^(mu2+4);
M6calc1=n01.*gamma(mu1+7)./lambda1.^(mu1+7);
M6calc2=n02.*gamma(mu2+7)./lambda2.^(mu2+7);

diff1=abs(M6-M6calc1);
diff2=abs(M6-M6calc2);
mufinal=zeros(r,1);
for i=1:r
    if diff1(i)>diff2(i)
        mufinal(i)=mu2(i);
    else
        mufinal(i)=mu1(i);
    end
end
% recalculate lambda
lambdafinal=gamma(mufinal+4)./gamma(mufinal+3).*M2./M3;
% recalculate n0
n0final=M0.*lambdafinal.^(mufinal+1)./gamma(mufinal+1);

