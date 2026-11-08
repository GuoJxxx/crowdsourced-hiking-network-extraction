function state=tb_restore()
% A failed load stays inside this function and cannot partially mutate base.
names={'checkpoint_current.mat','checkpoint_previous.mat'};
found=false;
for k=1:numel(names)
    if exist(names{k},'file')~=2, continue; end
    found=true;
    try
        state=load(names{k});
        assert(isfield(state,'tb_phase') && isfield(state,'tb_loop_next'));
        if k==2 && exist(names{1},'file')==2
            % Never rotate a known-corrupt current over the valid previous.
            % Keep the failed file for diagnosis; a power loss during the next
            % snapshot still leaves checkpoint_previous.mat readable.
            quarantine=[tempname(pwd) '.corrupt.mat'];
            movefile(names{1},quarantine);
        end
        fprintf('[resume] %s phase=%d\n',names{k},state.tb_phase);
        return;
    catch exc
        fprintf(2,'[resume] cannot load %s: %s\n',names{k},exc.message);
    end
end
if found, error('TraceBundle:InvalidCheckpoint','No valid checkpoint remains'); end
state=struct('tb_phase',1,'tb_loop_next',1,'tb_loop_active',0,'tb_loop_values',[]);
end
