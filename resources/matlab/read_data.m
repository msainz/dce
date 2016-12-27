clc
close all
clear


basename = '/home/love/dce/results/';


foldername = 'Pinter_M=50_W=hasting_weights_nodes_10_graph_1_gamma=0.95_lb=-50.0_ub=50.0_Sinit=100.0_Sinc=false';
% foldername = '20161227_020834_Pinter_M=50_W=hasting_weights_nodes_10_graph_1_gamma=0.95_lb=-50.0_ub=50.0_Sinit=100.0_Sinc=false';
% foldername = 'Griewank_M=50_W=hasting_weights_nodes_10_graph_1_gamma=0.95_lb=-50.0_ub=50.0_Sinit=100.0_Sinc=false';

f = dir(basename);
num_dirs = size(f);
optsol_err = [];
optval_err = [];
num_it = 0;
for i = 1 : num_dirs
    k = strfind(f(i).name,foldername);
    if ~isempty(k)
        FolderName = [basename, f(i).name, '/'];
        [optsol_err_dir, optval_err_dir, num_it_dir] = get_error_values_from_file (FolderName);
        [optsol_err, optval_err, num_it] = ...
            include_values(optsol_err_dir, optval_err_dir, num_it_dir, ...
                optsol_err, optval_err, num_it);
        semilogy(1:num_it,[optval_err, optsol_err])
        drawnow
    end
    i
end





