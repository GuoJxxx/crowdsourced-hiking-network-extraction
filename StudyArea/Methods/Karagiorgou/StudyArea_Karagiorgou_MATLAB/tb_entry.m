% Dedicated MATLAB process; restore state before executing generated scripts.
clear;
addpath(genpath('libraries'));
tb_expected_run_id='';
if exist('../identity.json','file')==2
    tb_spec=jsondecode(fileread('../identity.json'));
    tb_expected_run_id=tb_spec.run_id;
end
tb_state=tb_restore();
if ~isempty(tb_expected_run_id)
    if tb_state.tb_phase>1
        assert(isfield(tb_state,'tb_run_id') && strcmp(tb_state.tb_run_id,tb_expected_run_id), ...
               'Checkpoint belongs to different code or inputs');
    end
    tb_state.tb_run_id=tb_expected_run_id;
end
tb_names=fieldnames(tb_state);
for tb_name_index=1:numel(tb_names)
    assignin('base',tb_names{tb_name_index},tb_state.(tb_names{tb_name_index}));
end
clear tb_state tb_names tb_name_index tb_spec tb_expected_run_id;
diary('matlab_live.log');
try
    run('extraction_resume.m');
    run('bundle_resume.m');
    if exist('tb_run_id','var') && ~isempty(tb_run_id)
        tb_marker=struct('run_id',tb_run_id,'phase',tb_phase, ...
                        'vertices',size(n_nodes,1),'edges',size(n_edges,1));
        tb_marker_file=fopen('completion.json.tmp','w');
        assert(tb_marker_file>=0,'Cannot write completion marker');
        fprintf(tb_marker_file,'%s\n',jsonencode(tb_marker)); fclose(tb_marker_file);
        movefile('completion.json.tmp','completion.json','f');
        clear tb_marker tb_marker_file;
    end
    fprintf('[completed] %s\n',datestr(now,31));
catch tb_failure
    fprintf(2,'%s\n',getReport(tb_failure,'extended','hyperlinks','off'));
    diary off;
    rethrow(tb_failure);
end
diary off;
