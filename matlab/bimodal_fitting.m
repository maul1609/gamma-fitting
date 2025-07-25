% bimodal fitting
if (~exist('fittingFlag','var'))
    fittingFlag=true;
end
% for i =1:length(Z(:))
% [n01(i),n02(i),mu1(i),mu2(i),lam1(i),lam2(i),a,b,c,d,e,f]=CalcParams([X(i),Y(i),Z(i)],M0(325),M2(325),M3(325),M6(325));
% end
warning off;
[r,c1]=size(now_data_DP);
if(exist('dwidthCDP','var'))
    dedge=dcentreCDP'-0.5.*dwidthCDP';
    dedge=[dedge,dcentreCDP(end)+0.5.*dwidthCDP(end)]
else
    dedge=[2,d_size];
end

% store all the moments
M0=now_total_Nd.*1e6;
M2=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^2,[r, 1]),2);
M3=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^3,[r, 1]),2);
M6=sum(now_data_DP.*1e6.*repmat((d_size./1e6).^6,[r, 1]),2);

% arrays for parameters
if fittingFlag
    mu1=zeros(r,1);
    mu2=zeros(r,1);
    lambda1=zeros(r,1);
    lambda2=zeros(r,1);
    n01=zeros(r,1);
    n02=zeros(r,1);
end

% options1=optimset('display','off');
% loop over all times
[ix,iy,iz]=meshgrid(0.05:0.05:0.95);
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
c.power=2;c.powerPlot=0;

% init1=minis;

method='curvefit';
howmany='all';

Xstore=zeros(r,3);
if fittingFlag
    if strcmp(method, 'curvefit') | strcmp(method, 'lsqnonlin')
        n01=zeros(r,1);n02=zeros(r,1);
        mu1=zeros(r,1);mu2=zeros(r,1);
        lam1=zeros(r,1);lam2=zeros(r,1);
        M0_1=zeros(r,1);M0_2=zeros(r,1);
        M2_1=zeros(r,1);M2_2=zeros(r,1);
        M3_1=zeros(r,1);M3_2=zeros(r,1);
    else
        n01e=zeros(r,1);
        mu1e=zeros(r,1);
        lam1e=zeros(r,1);
        M0_1e=zeros(r,1);
        M2_1e=zeros(r,1);
        M3_1e=zeros(r,1);
    end
end

switch howmany
    case 'one'
        start=3170;
        end1=start;
    case 'all'
        start=1;
        end1=r;
    otherwise
        start=1;
        end1=r;
end

if fittingFlag
    for i=start:end1
        % store moments in c
        c.M0=M0(i);
        c.M2=M2(i);
        c.M3=M3(i);
        c.M6=M6(i);
        j=find(now_data_DP(i,:)>0);
        j=1:30;length(now_data_DP(i,:));
        switch method
            case 'easy'
                [n01e(i),mu1e(i),lam1e(i),M0_1e(i),M2_1e(i),M3_1e(i)]=...
                    CalcParamsEasy(M0(i),M2(i),M3(i),M6(i));
            case 'curvefit'
                % minimize
                % try
                    val=max(now_data_DP(i,:),0) ...
                        .*1e12./diff(dedge).*(d_size./1e6).^c.power;
                    %dat=real(log10(val));
                    dat=val;
                    RESNORMS=[];
                    Xs=[];
                    for k=1:length(ix(:))
                        RESIDUAL=PSDfit([ix(k),iy(k),iz(k)],d_size(j)./1e6,c)...
                            -dat(j);
                        RESNORM=norm(RESIDUAL);
                        RESNORMS=[RESNORMS;RESNORM];
                    end
                    ind=find(min(RESNORMS)==RESNORMS);
    
                    [X,RESNORM,RESIDUAL,EXITFLAG] = ...
                        lsqcurvefit(@(x,xdata) PSDfit(x,xdata,c),...
                        [ix(ind(1)),iy(ind(1)),iz(ind(1))], ...
                        d_size(j)./1e6, dat(j),...
                        minis,maxis,options1);
                    %X=[ix(ind) iy(ind) iz(ind)];
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
        disp([num2str(i),' of ',num2str(end1-start+1)]);
    end
end
% calculate the PSD
PSD=now_data_DP(:,:).*1e12./repmat(diff(dedge),[r,1]);
if fittingFlag
    % calculate the PSD from fit
    PSDcalc1=zeros(size(PSD));
    PSDcalc2=zeros(size(PSD));
    PSDcalc=zeros(size(PSD));
end

switch howmany
    case 'all'
        switch method
            case {'curvefit','lsqnonlin'}
                if fittingFlag
                    % calculate the average diameter of both modes
                    D1 = (mu1+1)./lam1;
                    D2 = (mu2+1)./lam2;
            
                    % gamma(mu+3)=(mu+2)*gamma(mu+2)
    
                    for i=1:r
                        % swap
                        [D1(i),D2(i),n01(i),n02(i),...
                            lam1(i),lam2(i),mu1(i),mu2(i),...
                            M0_1(i),M0_2(i),M2_1(i),M2_2(i),...
                            M3_1(i),M3_2(i)]= ...
                            swap(D1(i),D2(i),n01(i),n02(i),...
                            lam1(i),lam2(i),mu1(i),mu2(i),...
                            M0_1(i),M0_2(i),M2_1(i),M2_2(i),...
                            M3_1(i),M3_2(i));
    
                        PSDcalc1(i,:)=n01(i).*(d_size./1e6).^mu1(i).* ...
                            exp(-lam1(i).*d_size./1e6);
                        PSDcalc2(i,:)=n02(i).*(d_size./1e6).^mu2(i).* ...
                            exp(-lam2(i).*d_size./1e6);
                        PSDcalc(i,:)=PSDcalc1(i,:)+PSDcalc2(i,:);
                    end
                    % calculate average size
                    Dav=(n01.*gamma(mu1+2)./lam1.^(mu1+2)+ ...
                        n02.*gamma(mu2+2)./lam2.^(mu2+2)) ./ ...
                        (M0);
                    % calculate variance
                    var1=n01.*gamma(mu1+3)./lam1.^(mu1+3);
                    var2=n02.*gamma(mu2+3)./lam2.^(mu2+3);
                    vartot=(var1+var2)./ M0 - Dav.^2;
                    % dispersion
                    disptot = sqrt(vartot)./Dav;
                    betat=(1+2.*disptot.^2).^(2/3)./(1+disptot.^2).^(1/3);
                end
                Ddat=sum(now_data_DP.*repmat(d_size./1e6,[r,1]),2)./ ...
                    sum(now_data_DP,2);

                % expected value of D^2 minus expected value of (D)^2
                vardat=sum(now_data_DP.*repmat(d_size./1e6,[r,1]).^2,2)./ ...
                    sum(now_data_DP,2)-Ddat.^2;
                % dispersion
                dispdat = sqrt(vardat)./Ddat;
                beta1=(1+2.*dispdat.^2).^(2/3)./(1+dispdat.^2).^(1/3);
                F1=M2.^3./(M3.^2.*M0);
                if fittingFlag
                    figure;
                    subplot(211)
                    pcolor(real(log10(PSD'.*repmat(d_size.^c.powerPlot,[r,1])')));shading flat
                    c1=caxis;
                    subplot(212)
                    pcolor(real(log10(PSDcalc'.*repmat(d_size.^c.powerPlot,[r,1])')));shading flat
                    caxis(c1);
                end
            case 'easy'
                if fittingFlag
                    for i=1:r
                        PSDcalc1(i,:)=n01e(i).*(d_size./1e6).^mu1e(i).* ...
                            exp(-lam1e(i).*d_size./1e6);
                        PSDcalc(i,:)=PSDcalc1(i,:);
                    end
                    % calculate average size
                    Dave=(n01e.*gamma(mu1e+2)./lam1e.^(mu1e+2)) ./ ...
                        (M0_1e);
                    % calculate variance
                    var1e=M2_1e;
                    vartote=max((var1e)./ M0_1e - Dave.^2,0);
                    % dispersion
                    disptote = sqrt(vartote)./Dave;
                    betate=(2+2.*disptote.^2).^(2/5)./(1+disptote.^2).^(1/5);
                end
                Ddat=sum(now_data_DP.*repmat(d_size./1e6,[r,1]),2)./ ...
                    sum(now_data_DP,2);

                % expected value of D^2 minus expected value of (D)^2
                vardat=sum(now_data_DP.*repmat(d_size./1e6,[r,1]).^2,2)./ ...
                    sum(now_data_DP,2)-Ddat.^2;

                % dispersion
                dispdat = sqrt(vardat)./Ddat;
                beta1=(2+2.*dispdat.^2).^(2/5)./(1+dispdat.^2).^(1/5);
                F1=M2.^3./(M3.^2.*M0);
                if fittingFlag
                    figure;
                    subplot(211)
                    pcolor(real(log10(PSD'.*repmat(d_size.^0,[r,1])')));shading flat
                    c1=caxis;
                    subplot(212)
                    pcolor(real(log10(PSDcalc'.*repmat(d_size.^0,[r,1])')));shading flat
                    caxis(c1);
                end
            otherwise
                disp('Unknown method')
        end
    case 'one'
        if fittingFlag
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
end

% plot(F1,(1+2.*dispdat.^2).^(2/3)./(1+dispdat.^2).^(1./3),'.')
% https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2003GL017192
% scatter(M3,beta1,20,F1,'filled')

function F=PSDfit(x,xdata,c)

% x contains the fraction of M0, M2, M3 in the distribution
M0=c.M0;
M2=c.M2;
M3=c.M3;
M6=c.M6;

x=real(x);
[n01,n02,mu1,mu2,lam1,lam2,...
    M0_1,M2_1,M3_1,M0_2,M2_2,M3_2]=CalcParams(x,M0,M2,M3,M6);

%F=real(log10(max(n01,0).*xdata.^mu1.*exp(-lam1.*xdata).*xdata.^c.power+...
%     max(n02,0).*xdata.^mu2.*exp(-lam2.*xdata).*xdata.^c.power));
F=real((max(n01,0).*xdata.^mu1.*exp(-lam1.*xdata).*xdata.^c.power+...
    max(n02,0).*xdata.^mu2.*exp(-lam2.*xdata).*xdata.^c.power));

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
mur1=max(-1,real(roots([(1-F1),(3-6.*F1),(2-9.*F1)])));
% mur1=real(roots([(1-F1),(3-6.*F1),(2-9.*F1)]));
if length(mur1)==1
    mur1=[mur1,mur1];
end
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
mur2=max(real(roots([(1-F2),(3-6.*F2),(2-9.*F2)])),-1);
if length(mur2)==1
    mur2=[mur2,mur2];
end
%mur2=real(roots([(1-F2),(3-6.*F2),(2-9.*F2)]));
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
mur1=max(-1,real(roots([(1-F1),(3-6.*F1),(2-9.*F1)])));
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

function [D1,D2,n01,n02,lam1,lam2,mu1,mu2,...
                        M0_1,M0_2,M2_1,M2_2,...
                        M3_1,M3_2]= ...
                        swap(D1,D2,n01,n02,...
                        lam1,lam2,mu1,mu2,...
                        M0_1,M0_2,M2_1,M2_2,...
                        M3_1,M3_2)

if(D1 > D2)
    dtmp=D1;
    D1=D2;
    D2=dtmp;
    % 
    dtmp=n01;
    n01=n02;
    n02=dtmp;
    % 
    dtmp=lam1;
    lam1=lam2;
    lam2=dtmp;
    % 
    dtmp=mu1;
    mu1=mu2;
    mu2=dtmp;
    % 
    dtmp=M0_1;
    M0_1=M0_2;
    M0_2=dtmp;
    % 
    dtmp=M2_1;
    M2_1=M2_2;
    M2_2=dtmp;
    % 
    dtmp=M3_1;
    M3_1=M3_2;
    M3_2=dtmp;
end

end
