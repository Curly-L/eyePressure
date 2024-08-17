% 创建任意类型的分段函数的拟合类型，用做fittype进行曲线拟合

% x原始数据
% fit_y用于分段原始数据以拟合函数的横坐标值
% xDuration表示每段的拟合方式

function y = piecewiseLine(x,fit_y,segments,cross_x)
% PIECEWISELINE   A line made of several pieces
% that is not continuous.

    y = zeros(size(x));
    lineNum = length(segments)-1;

    for i = 1:length(x)
        for j = 1:lineNum
            if j == 1  % First segment
                if i >= segments(1) 
                    y(i) = feval(fit_y{j},x(i));
                    break;
                end
            elseif j == lineNum  % Last segment
                if i <= segments(end)
                    y(i) = feval(fit_y{j},x(i));
                    break;
                end
            else  % Intermediate segments
                if i > cross_x(j-1) && i <= cross_x(j)
                    y(i) = feval(fit_y{j},x(i));
                    break;
                end
            end
        end
    end
end