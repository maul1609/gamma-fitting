% bimodal fitting
% for i =1:length(Z(:))
% [n01(i),n02(i),mu1(i),mu2(i),lam1(i),lam2(i),a,b,c,d,e,f]=CalcParams([X(i),Y(i),Z(i)],M0(325),M2(325),M3(325),M6(325));
% end
warning off;
[r,c1]=size(now_data_DP);
dedge=[2,d_size];


% store all the moments
M0=now_total_Nd.*1e6;
M2=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^2,[r, 1]),2);
M3=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^3,[r, 1]),2);
M6=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^6,[r, 1]),2);

% arrays for parameters
mu1=zeros(r,1);
mu2=zeros(r,1);
lambda1=zeros(r,1);
lambda2=zeros(r,1);
n01=zeros(r,1);
n02=zeros(r,1);

% options1=optimset('display','off');
% loop over all times
[ix,iy,iz]=meshgrid(0.1:0.1:0.9);
init1=[0.3,0.6,0.85];
minis=[1.e-4,1e-4,1e-4];
maxis=1-minis;

% Algorithm: [ active-set | interior-point | interior-point-convex | levenberg-marquardt | ...
%                            sqp | trust-region-dogleg | trust-region-reflective ]
options1=optimoptions('lsqcurvefit',...
    'algorithm','interior-point','display','off');
% options1=optimoptions('lsqnonlin','MaxFunEvals',10000,'MaxIter',10000,...
%     'algorithm','interior-point','StepTolerance',1e-10,...
%     'OptimalityTolerance',1e-10,'TolFun',1e-8,'TypicalX',init1);
c.power=0;c.powerPlot=0;

% init1=minis;
Xstore=zeros(r,3);
n01=zeros(r,1);n02=zeros(r,1);
mu1=zeros(r,1);mu2=zeros(r,1);
lam1=zeros(r,1);lam2=zeros(r,1);
M0_1=zeros(r,1);M0_2=zeros(r,1);
M2_1=zeros(r,1);M2_2=zeros(r,1);
M3_1=zeros(r,1);M3_2=zeros(r,1);
method='curvefit';
howmany='all';

switch howmany
    case 'one'
        start=200;
        end1=start;
    case 'all'
        start=1;
        end1=r;
    otherwise
        start=1;
        end1=r;
end


for i=start:end1
    % store moments in c
    c.M0=M0(i);
    c.M2=M2(i);
    c.M3=M3(i);
    c.M6=M6(i);
    j=find(now_data_DP(i,:)>0);
    j=1:length(now_data_DP(i,:));
    switch method
        case 'easy'
            [n01(i),mu1(i),lam1(i),M0_1(i),M2_1(i),M3_1(i)]=...
                CalcParamsEasy(M0(i),M2(i),M3(i),M6(i));
        case 'curvefit'
            % minimize
            % try
                val=max(now_data_DP(i,:),1e-10) ...
                    .*1e12./diff(dedge).*(d_size./1e6).^c.power;
                dat=real(log10(val));
                dat=val;
                RESNORMS=[];
                Xs=[];
                for k=1:length(ix(:))
                    [X,RESNORM,RESIDUAL,EXITFLAG] = ...
                        lsqcurvefit(@(x,xdata) PSDfit(x,xdata,c),...
                        [ix(k),iy(k),iz(k)], ...
                        d_size(j)./1e6, dat(j),...
                        minis,maxis,options1);
                    Xs=[Xs;X];
                    RESNORMS=[RESNORMS;RESNORM];
                end
                ind=find(min(RESNORMS)==RESNORMS);
                X=Xs(ind(1),:);
                Xstore(i,:)=real(X);

                [n01(i),n02(i),mu1(i),mu2(i),lam1(i),lam2(i), ...
                    M0_1(i),M2_1(i),M3_1(i),M0_2(i),M2_2(i),M3_2(i)]=...
                    CalcParams(X,M0(i),M2(i),M3(i),M6(i));
            % catch
            %    1==1; 
            % end
        case 'lsqnonlin'
            c.ydata=now_data_DP(i,:).*1e12./diff(dedge);
            c.ydata(j);
            c.xdata=d_size(j)./1e6;

            try
                X = lsqnonlin(@(x) PSDfit2(x,c),init1, ...
                    minis,maxis,options1);
                Xstore(i,:)=real(X);
            
                [n01(i),n02(i),mu1(i),mu2(i),lam1(i),lam2(i), ...
                    M0_1(i),M2_1(i),M3_1(i),M0_2(i),M2_2(i),M3_2(i)]=...
                    CalcParams(X,M0(i),M2(i),M3(i),M6(i));
            catch
                1==1;
            end
        otherwise
            disp('unknown')
    end
    disp(num2str(i));
end

% calculate the PSD
PSD=now_data_DP(:,:).*1e12./repmat(diff(dedge),[r,1]);
% calculate the PSD from fit
PSDcalc1=zeros(size(PSD));
PSDcalc2=zeros(size(PSD));
PSDcalc=zeros(size(PSD));

switch howmany
    case 'all'
        switch method
            case {'curvefit','lsqnonlin'}
                % calculate the average diameter of both modes
                D1 = gamma(mu1+3)./lam1.^(mu1+3) ./ (gamma(mu1+2)./lam1.^(mu1+2));
                D2 = gamma(mu2+3)./lam2.^(mu2+3) ./ (gamma(mu2+2)./lam2.^(mu2+2));
        
                for i=1:r
                    PSDcalc1(i,:)=n01(i).*(d_size./1e6).^mu1(i).* ...
                        exp(-lam1(i).*d_size./1e6);
                    PSDcalc2(i,:)=n02(i).*(d_size./1e6).^mu2(i).* ...
                        exp(-lam2(i).*d_size./1e6);
                    PSDcalc(i,:)=PSDcalc1(i,:)+PSDcalc2(i,:);
                end
        
                figure;
                subplot(211)
                pcolor(real(log10(PSD'.*repmat(d_size.^c.powerPlot,[r,1])')));shading flat
                c1=caxis;
                subplot(212)
                pcolor(real(log10(PSDcalc'.*repmat(d_size.^c.powerPlot,[r,1])')));shading flat
                caxis(c1);
            case 'easy'
                for i=1:r
                    PSDcalc1(i,:)=n01(i).*(d_size./1e6).^mu1(i).* ...
                        exp(-lam1(i).*d_size./1e6);
                    PSDcalc(i,:)=PSDcalc1(i,:);
                end
        
                figure;
                subplot(211)
                pcolor(real(log10(PSD'.*repmat(d_size.^0,[r,1])')));shading flat
                c1=caxis;
                subplot(212)
                pcolor(real(log10(PSDcalc'.*repmat(d_size.^0,[r,1])')));shading flat
                caxis(c1);
            otherwise
                disp('Unknown method')
        end
    case 'one'
        for i=start:end1
            PSDcalc1(i,:)=n01(i).*(d_size./1e6).^mu1(i).* ...
                exp(-lam1(i).*d_size./1e6);
            PSDcalc2(i,:)=n02(i).*(d_size./1e6).^mu2(i).* ...
                exp(-lam2(i).*d_size./1e6);
            PSDcalc(i,:)=PSDcalc1(i,:)+PSDcalc2(i,:);
        end
        figure;
        plot(d_size./1e6,PSD(start,:).*(d_size./1e6).^c.powerPlot);hold on;
        plot(d_size./1e6,PSDcalc(start,:).*(d_size./1e6).^c.powerPlot);

end

function F=PSDfit(x,xdata,c)

% x contains the fraction of M0, M2, M3 in the distribution
M0=c.M0;
M2=c.M2;
M3=c.M3;
M6=c.M6;

[n01,n02,mu1,mu2,lam1,lam2,...
    M0_1,M2_1,M3_1,M0_2,M2_2,M3_2]=CalcParams(x,M0,M2,M3,M6);

F=real(log10(max(n01,0).*xdata.^mu1.*exp(-lam1.*xdata).*xdata.^c.power+...
    max(n02,0).*xdata.^mu2.*exp(-lam2.*xdata).*xdata.^c.power));
% F=real((max(n01,0).*xdata.^mu1.*exp(-lam1.*xdata).*xdata.^c.power+...
%     max(n02,0).*xdata.^mu2.*exp(-lam2.*xdata).*xdata.^c.power));

end

function F=PSDfit2(x,c)

% x contains the fraction of M0, M2, M3 in the distribution
M0=c.M0;
M2=c.M2;
M3=c.M3;
M6=c.M6;
xdata=c.xdata;
ydata=c.ydata;
[n01,n02,mu1,mu2,lam1,lam2,...
    M0_1,M2_1,M3_1,M0_2,M2_2,M3_2]=CalcParams(x,M0,M2,M3,M6);

F=(real(max(n01,0)).*xdata.^mu1.*exp(-lam1.*xdata).*xdata.^c.power+...
    real(max(n02,0)).*xdata.^mu2.*exp(-lam2.*xdata).*xdata.^c.power ...
    -ydata.*xdata.^c.power);

end


function [n01,n02,mu1,mu2,lam1,lam2,...
    M0_1,M2_1,M3_1,M0_2,M2_2,M3_2]=CalcParams(x,M0,M2,M3,M6)

% moments in mode 1
M0_1=M0*real(x(1));
M2_1=M2*real(x(2));
M3_1=M3*real(x(3));

% moments in mode 2
M0_2=M0*(1.0-real(x(1)));
M2_2=M2*(1.0-real(x(2)));
M3_2=M3*(1.0-real(x(3)));

% F=M(2)^3/(M(3)^2*M(0))
% (1-F)*mu^2+(3-6F)*mu+(2-9F)=0

% lam=gamma(mu+4)/gamma(mu+3)*M(2)/M(3)
% n0=M(0)*lam^(mu+1)/gamma(mu+1)

% for each mode, calculate the parameters
% MODE1++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% F calc
F1=M2_1.^3./(M3_1.^2.*M0_1);
% find the roots
mur1=max(0,real(roots([(1-F1),(3-6.*F1),(2-9.*F1)])));
% calculate lambda
lambda1_1=gamma(mur1(1)+4)./gamma(mur1(1)+3).*M2_1./M3_1;
lambda2_1=gamma(mur1(2)+4)./gamma(mur1(2)+3).*M2_1./M3_1;
% calculate n0
n01_1=M0_1.*lambda1_1.^(mur1(1)+1)./gamma(mur1(1)+1);
n02_1=M0_1.*lambda2_1.^(mur1(2)+1)./gamma(mur1(2)+1);

% recalculating M0, M2, M3
M0calc1_1=n01_1.*gamma(mur1(1)+1)./lambda1_1.^(mur1(1)+1);
M0calc2_1=n02_1.*gamma(mur1(2)+1)./lambda2_1.^(mur1(2)+1);
M2calc1_1=n01_1.*gamma(mur1(1)+3)./lambda1_1.^(mur1(1)+3);
M2calc2_1=n02_1.*gamma(mur1(2)+3)./lambda2_1.^(mur1(2)+3);
M3calc1_1=n01_1.*gamma(mur1(1)+4)./lambda1_1.^(mur1(1)+4);
M3calc2_1=n02_1.*gamma(mur1(2)+4)./lambda2_1.^(mur1(2)+4);
M6calc1_1=n01_1.*gamma(mur1(1)+7)./lambda1_1.^(mur1(1)+7);
M6calc2_1=n02_1.*gamma(mur1(2)+7)./lambda2_1.^(mur1(2)+7);
%--------------------------------------------------------------------------

% MODE2++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% F calc
F2=M2_2.^3./(M3_2.^2.*M0_2);
% find the roots
mur2=max(real(roots([(1-F2),(3-6.*F2),(2-9.*F2)])),0);
% calculate lambda
lambda1_2=gamma(mur2(1)+4)./gamma(mur2(1)+3).*M2_2./M3_2;
lambda2_2=gamma(mur2(2)+4)./gamma(mur2(2)+3).*M2_2./M3_2;
% calculate n0
n01_2=M0_2.*lambda1_2.^(mur2(1)+1)./gamma(mur2(1)+1);
n02_2=M0_2.*lambda2_2.^(mur2(2)+1)./gamma(mur2(2)+1);


% recalculating M0, M2, M3
M0calc1_2=n01_2.*gamma(mur2(1)+1)./lambda1_2.^(mur2(1)+1);
M0calc2_2=n02_2.*gamma(mur2(2)+1)./lambda2_2.^(mur2(2)+1);
M2calc1_2=n01_2.*gamma(mur2(1)+3)./lambda1_2.^(mur2(1)+3);
M2calc2_2=n02_2.*gamma(mur2(2)+3)./lambda2_2.^(mur2(2)+3);
M3calc1_2=n01_2.*gamma(mur2(1)+4)./lambda1_2.^(mur2(1)+4);
M3calc2_2=n02_2.*gamma(mur2(2)+4)./lambda2_2.^(mur2(2)+4);
M6calc1_2=n01_2.*gamma(mur2(1)+7)./lambda1_2.^(mur2(1)+7);
M6calc2_2=n02_2.*gamma(mur2(2)+7)./lambda2_2.^(mur2(2)+7);
%--------------------------------------------------------------------------

% now choose the combinations 1,1; 1,2; 2,1; 2,2 that best match M6
M6calc_11 = M6calc1_1+M6calc1_2;
M6calc_12 = M6calc1_1+M6calc2_2;
M6calc_21 = M6calc2_1+M6calc1_2;
M6calc_22 = M6calc2_1+M6calc2_2;
% arr=[1,1;1,2;2,1;2,2];

M0calc_11 = M0calc1_1+M0calc1_2;
M0calc_12 = M0calc1_1+M0calc2_2;
M0calc_21 = M0calc2_1+M0calc1_2;
M0calc_22 = M0calc2_1+M0calc2_2;

M2calc_11 = M2calc1_1+M2calc1_2;
M2calc_12 = M2calc1_1+M2calc2_2;
M2calc_21 = M2calc2_1+M2calc1_2;
M2calc_22 = M2calc2_1+M2calc2_2;


M3calc_11 = M3calc1_1+M3calc1_2;
M3calc_12 = M3calc1_1+M3calc2_2;
M3calc_21 = M3calc2_1+M3calc1_2;
M3calc_22 = M3calc2_1+M3calc2_2;

diffs=abs([M6-M6calc_11; M6-M6calc_12; M6-M6calc_21; M6-M6calc_22]);
n0=[n01_1,n01_2;n01_1,n02_2;n02_1,n01_2;n02_1,n02_2];
lam=[lambda1_1,lambda1_2;lambda1_1,lambda2_2;lambda2_1,lambda1_2;lambda2_1,lambda2_2];
mux=[mur1(1),mur2(1);mur1(1),mur2(2);mur1(2),mur2(1);mur1(2),mur2(2)];

% find the correct combination, based on M6 best match
ind=find(diffs==min(diffs));ind=ind(1);

n01=n0(ind,1);
n02=n0(ind,2);
mu1=mux(ind,1);
mu2=mux(ind,2);
lam1=lam(ind,1);
lam2=lam(ind,2);
end


function [n01,mu1,lam1,M0calc,M2calc,M3calc]=CalcParamsEasy(M0,M2,M3,M6)

% F=M(2)^3/(M(3)^2*M(0))
% (1-F)*mu^2+(3-6F)*mu+(2-9F)=0

% lam=gamma(mu+4)/gamma(mu+3)*M(2)/M(3)
% n0=M(0)*lam^(mu+1)/gamma(mu+1)

% for each mode, calculate the parameters
% MODE1++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% F calc
F1=M2.^3./(M3.^2.*M0);
% find the roots
mur1=real(roots([(1-F1),(3-6.*F1),(2-9.*F1)]));
% calculate lambda
lambda1=gamma(mur1(1)+4)./gamma(mur1(1)+3).*M2./M3;
lambda2=gamma(mur1(2)+4)./gamma(mur1(2)+3).*M2./M3;
% calculate n0
n01=M0.*lambda1.^(mur1(1)+1)./gamma(mur1(1)+1);
n02=M0.*lambda2.^(mur1(2)+1)./gamma(mur1(2)+1);

% recalculating M0, M2, M3
M0calc1=n01.*gamma(mur1(1)+1)./lambda1.^(mur1(1)+1);
M0calc2=n02.*gamma(mur1(2)+1)./lambda2.^(mur1(2)+1);
M2calc1=n01.*gamma(mur1(1)+3)./lambda1.^(mur1(1)+3);
M2calc2=n02.*gamma(mur1(2)+3)./lambda2.^(mur1(2)+3);
M3calc1=n01.*gamma(mur1(1)+4)./lambda1.^(mur1(1)+4);
M3calc2=n02.*gamma(mur1(2)+4)./lambda2.^(mur1(2)+4);
M6calc1=n01.*gamma(mur1(1)+7)./lambda1.^(mur1(1)+7);
M6calc2=n02.*gamma(mur1(2)+7)./lambda2.^(mur1(2)+7);
%--------------------------------------------------------------------------


% now choose the combinations 1,1; 1,2; 2,1; 2,2 that best match M6
diffs=abs([M6-M6calc1; M6-M6calc2]);
n0=[n01,n02];
lam=[lambda1,lambda2];
mux=[mur1(1),mur1(2)];

% find the correct combination, based on M6 best match
ind=find(diffs==min(diffs));ind=ind(1);

n01=n0(ind);
mu1=mux(ind);
lam1=lam(ind);

M0calc=n01.*gamma(mu1+1)./lam1.^(mu1+1);
M2calc=n01.*gamma(mu1+3)./lam1.^(mu1+3);
M3calc=n01.*gamma(mu1+4)./lam1.^(mu1+4);
end
