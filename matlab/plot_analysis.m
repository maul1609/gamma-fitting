figure

project='vocals';

switch project
    case 'vocals'
        load vocals.mat
    case 'mphase'
        load mphase.mat
    case 'dcmex'
        load dcmex.mat
    otherwise
        disp('no project with this name defined')
end

t=tiledlayout(4,4,'padding','none','tilespacing','compact') 

for i=1:9
    time=ncread([path,files1{i}],'Time');
    temp=ncread([path,files1{i}],'TAT_DI_R');
    nexttile

    tempDisp=interp1(time,nanmean(temp,1)-273.15,double(fit_data(i).timeCDP));

    Ddat=sum(fit_data(i).PSD.*repmat(fit_data(i).dcentreCDP'./1e6,[length(fit_data(i).timeCDP),1]),2)./ ...
        sum(fit_data(i).PSD,2);

    % ind=find( (fit_data(i).D1>1e-6));
    % plot(fit_data(i).M3(ind),tempDisp(ind),'.');hold on;
    % ind=find( (abs(fit_data(i).D1-fit_data(i).D2)>8e-6) & (fit_data(i).M0_1>5e6) & (fit_data(i).M0_2>5e6) & (fit_data(i).D1>1e-6));
    % plot(fit_data(i).M3(ind),tempDisp(ind),'.');hold on;
    
    [med, centers, lower_err, upper_err]=calcErrorBars(tempDisp,fit_data(i).dispdat.*Ddat);
    errorbar(med, centers, lower_err, upper_err, 'horizontal', 'o', ...
        'CapSize', 5, 'MarkerSize', 6, 'LineWidth', 1.2);hold on;
    [med, centers, lower_err, upper_err]=calcErrorBars(tempDisp,Ddat);
    errorbar(med, centers, lower_err, upper_err, 'horizontal', 'o', ...
        'CapSize', 5, 'MarkerSize', 6, 'LineWidth', 1.2);hold on;

    hold on;ylim([-30,10]);
    set(gca,'YDir','reverse')
    xlim([0 30e-6])
    text(0.1,0.9,strrep(strrep(files1{i},'core_faam_',''),'_',' '),'units','normalized')
end

title(t,'Rates vs temperature DCMEX')
xlabel(t,'P (m^{-3} s^{-1})')
ylabel(t,'T (\circ C)')


function [med, centers, lower_err, upper_err]=calcErrorBars(T,N)

% Bin temperature into 2.5°C intervals
binWidth = 2;
edges = min(T):binWidth:max(T);
centers = edges(1:end-1) + binWidth/2;

% Initialize
med = NaN(size(centers));
q25 = NaN(size(centers));
q75 = NaN(size(centers));

% Compute statistics per bin
for i = 1:length(centers)
    inBin = T >= edges(i) & T < edges(i+1);
    if any(inBin)
        data = N(inBin);
        med(i) = median(data);
        q25(i) = quantile(data, 0.25);
        q75(i) = quantile(data, 0.75);
    end
end

% Error bars: horizontal
lower_err = med - q25;
upper_err = q75 - med;

end