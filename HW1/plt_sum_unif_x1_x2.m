function [output] = plt_sum_unif_x1_x2(x1,x2)
%plt_sum_unif_x1_x2 would compute y = x1+x2 and plot their graphs
% This function is for question 5

subplot(3,1,1);
% get the range and apply it to the plot function
x1_range = get_range(x1, length(x1));
plot(x1_range, x1);

x2_range = get_range(x2, length(x2));
subplot(3,1,2);
plot(x2_range, x2);

subplot(3,1,3);
y = convn(x1, x2);
y_range = get_range(y, length(y));
plot(y_range, y);

output = y;
end

