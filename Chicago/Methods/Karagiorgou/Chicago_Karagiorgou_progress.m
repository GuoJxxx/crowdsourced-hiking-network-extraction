function Chicago_Karagiorgou_progress(phase,completed,total,mode)
% Report actual work counts from the reference script's current outer loop.
% All persistent values belong to this reporting function, not the algorithm.
persistent activePhase lastPercent initialTotal currentTotal lastTick

changed = isempty(activePhase) || ~strcmp(activePhase,phase) || strcmp(mode,'start');
if changed
    activePhase = phase;
    initialTotal = max(0,floor(double(total)));
    currentTotal = initialTotal;
    lastPercent = -1;
    lastTick = tic;
end
if strcmp(mode,'remaining')
    completed = initialTotal - max(0,double(completed));
    total = initialTotal;
elseif strcmp(mode,'finish')
    total = currentTotal;
    completed = total;
end
total = max(0,floor(double(total)));
completed = max(0,min(total,floor(double(completed))));
currentTotal = total;
if total == 0
    percent = 100;
else
    percent = floor(100*completed/total);
end
if changed || percent ~= lastPercent || toc(lastTick) >= 15
    fprintf('[PROGRESS] {"phase":"%s","completed":%.0f,"total":%.0f,"percent":%.0f}\n', ...
        phase,completed,total,percent);
    lastPercent = percent;
    lastTick = tic;
end
end
