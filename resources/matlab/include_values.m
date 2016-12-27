function [optsol_err, optval_err, num_it] = ...
    include_values(optsol_err_dir, optval_err_dir, num_it_dir, ...
        optsol_err, optval_err, num_it)
    

if num_it > num_it_dir && num_it_dir > 0
    num_cols = size(optsol_err_dir, 2);
    optsol_err_dir = [optsol_err_dir; nan(num_it - num_it_dir, num_cols)];
    optval_err_dir = [optval_err_dir; nan(num_it - num_it_dir, num_cols)];
elseif num_it > 0 && num_it_dir > 0
    num_cols = size(optsol_err, 2);
    optsol_err = [optsol_err; nan(num_it_dir - num_it, num_cols)];
    optval_err = [optval_err; nan(num_it_dir - num_it, num_cols)];
    num_it = num_it_dir;
elseif num_it_dir > 0
    num_it = num_it_dir;
end
if num_it_dir > 0
    optsol_err = [optsol_err, optsol_err_dir];
    optval_err = [optval_err, optval_err_dir];
else
    warning(' num_it_dir = 0')
end


