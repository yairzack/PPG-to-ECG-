% convert the data from RRest_synth_generator.m to csv format similar to
% that used in the BIDMC dataset

data = load("Regenerated\RRSYNTHdata.mat");
data = data.data;
fs = 125; % nominal sampling frequency of 125Hz

for i = 1:length(data)
    hr = data(i).ref.params.hr.v;
    ppg = data(i).ppg.v';
    ecgII = data(i).ekg.v';
    time = linspace(0,length(ppg)/fs,length(ppg))';
    temp = table(time,ppg,ecgII,'VariableNames',{'Time [s]',' II',' PLETH'});
    idx = compose("%02.0f",i);
    hrText = compose("%02.0f",hr);
    % filename = sprintf("Regenerated\\synthSub%s.csv",idx);
    filename = sprintf("Regenerated\\synthSubHR%s.csv",hrText);
    writetable(temp,filename);
end