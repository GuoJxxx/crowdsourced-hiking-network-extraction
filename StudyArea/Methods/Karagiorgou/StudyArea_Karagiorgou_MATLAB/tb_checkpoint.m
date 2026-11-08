function tb_checkpoint(force)
% Save the entire script workspace and next loop iteration in rotating files.
% Only completed iterations are captured. Atomic rename keeps a prior snapshot.
persistent last_saved last_progress save_count
if isempty(last_saved), last_saved=tic; last_progress=tic; save_count=0; end
if force || toc(last_progress)>=15
    phase=evalin('base','tb_phase');
    next=evalin('base','tb_loop_next');
    total=evalin('base','numel(tb_loop_values)');
    record=struct('time',datestr(now,31),'phase',phase,'next_iteration',next, ...
                  'loop_total',total,'checkpoint_count_this_session',save_count);
    for variable={'k','rlen','scounter','g'}
        name=variable{1};
        if evalin('base',['exist(''' name ''',''var'')'])
            value=evalin('base',name);
            if isnumeric(value) && isreal(value) && isscalar(value)
                record.(name)=value;
            end
        end
    end
    fid=fopen('progress.json.tmp','w');
    if fid<0, error('TraceBundle:ProgressWrite','Cannot write progress'); end
    fprintf(fid,'%s\n',jsonencode(record)); fclose(fid);
    movefile('progress.json.tmp','progress.json','f');
    fprintf('[progress] %s phase=%d next=%d total=%d\n',datestr(now,31),phase,next,total);
    last_progress=tic;
end
interval=str2double(getenv('TRACEBUNDLE_TEST_CHECKPOINT_SECONDS'));
if ~isfinite(interval), interval=60; end
if ~force && toc(last_saved)<interval, return; end
evalin('base',"save('checkpoint_pending.mat','-v7.3');");
if exist('checkpoint_current.mat','file')==2
    movefile('checkpoint_current.mat','checkpoint_previous.mat','f');
end
movefile('checkpoint_pending.mat','checkpoint_current.mat','f');
last_saved=tic; save_count=save_count+1;
% A controlled stop is used only by integration tests to exercise resume.
stop_after=str2double(getenv('TRACEBUNDLE_TEST_STOP_AFTER'));
if isfinite(stop_after) && save_count>=stop_after
    error('TraceBundle:TestStop','Controlled checkpoint test stop');
end
end
