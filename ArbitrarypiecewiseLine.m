% 根据原始数据求任意类型的分段函数参数（cfit类型）

% x，y原始数据
% segments用于分段原始数据以拟合函数的数据序号
% fts表示每段的拟合方式
function [fit_y, cross_x] = piecewiseLineFunc(x,y,segments,fts)

    linesNum = length(segments)-1;
    crossDotsNum = linesNum-1;
    fit_y = cell(linesNum,1);
    cross_x = zeros(crossDotsNum);

    % 根据输入的拟合类型求每段的拟合函数
    for i = 1:linesNum
        ft = fts{i};
        seg_x = x(segments(i):segments(i+1));
        seg_y = y(segments(i):segments(i+1));
        fit_y{i} = fit(seg_x,seg_y,ft);
    end
    
    % 求阶段函数交点的横坐标
    for i = 1:crossDotsNum
        f1 = @(x) feval(fit_y{i},x);
        f2 = @(x) feval(fit_y{i+1},x);
        cross_x(i) = fzero(@(x) f1(x)-f2(x),segments(i+1));
    end
end

% 创建任意类型的分段函数的拟合类型，用piecewiseLineFunc得到的参数和横坐标进行曲线拟合

% x原始数据
% fit_y分段函数的每段拟合曲线参数
% segments用于分段原始数据以拟合函数的数据序号
% cross_x每段拟合曲线的交点

function y = piecewiseLine(x,fit_y,segments,cross_x)
% PIECEWISELINE   A line made of several pieces
% that is not continuous.

    y = zeros(size(x));
    lineNum = length(segments)-1;

    for i = 1:length(x)
        for j = 1:lineNum
            if j == lineNum  % Last segment
                if i <= segments(end)
                    y(i) = feval(fit_y{j},x(i));
                    break;
                end
            elseif j == 1  % First segment
                if i >= segments(1) && x(i)<cross_x(1)
                    y(i) = feval(fit_y{j},x(i));
                    break;
                end
            else  % Intermediate segments
                if x(i) > cross_x(j-1) && x(i) <= cross_x(j)
                    y(i) = feval(fit_y{j},x(i));
                    break;
                end
            end
        end
    end
end

%%
[data] = readmatrix('D:\蓝知医疗\嵌入式\youngs modulus calculation\youngs modulus.xls');
x = data(:,1);
y = data(:,2);

%%
    segments = [1, 2, 3, 13];
    fts = {'poly1', 'poly1', 'poly1'};

    [fit_y, cross_x] = piecewiseLineFunc(x,y,segments,fts);
    
    func = piecewiseLine(x,fit_y,segments,cross_x);

    figure;
    plot(x,func,'o',x,y,'*');
