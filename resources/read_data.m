clc
close all
clear


basename = '~/dce/results/';
foldername = '20161226_214127_Trigonometric_M=4_W=three-nodes_gamma=0.99_lb=-50.0_ub=50.0_Sinit=100.0_Sinc=false';

FolderName = [basename, foldername, '/'];

dir_files = dir(FolderName);
num_nodes = length(dir_files(not([dir_files.isdir])));
node_files = {};

j = 1;
for i = 1: size(dir_files, 1)
    if ~dir_files(i).isdir
        node_files{j} = [FolderName, dir_files(i).name];
        j = j+1;
    end
end


fileID = fopen(node_files{1},'r','n','ISO-8859-15');   
data = textscan(fileID, '%s %f %f %s %s', 'Delimiter',',');
fclose(fileID);
num_it = size(data{1},1);
optval_err = zeros(num_it, num_nodes);
optsol_err = zeros(num_it, num_nodes);

optval_err(:, 1) = data{2};
optsol_err(:, 1) = data{3};

for i = 2:num_nodes 
    fileID = fopen(node_files{i},'r','n','ISO-8859-15');   
    data = textscan(fileID, '%s %f %f %s %s', 'Delimiter',',');
    fclose(fileID);

    optval_err(:, i) = data{2};
    optsol_err(:, i) = data{3};
end

semilogy(1:num_it,[optval_err, optsol_err])
legend({'optval err', 'optsol err'})
