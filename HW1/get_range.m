function [range] = get_range(X, step)
%get_range Get the range of a two diminsional data and return it
%step shows how many steps we want for the data

max_x = ceil(max(max(X)));
min_x = floor(min(min(X)));
range = linspace(min_x, max_x, step);

end