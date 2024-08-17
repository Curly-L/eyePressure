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

%%
[data] = readmatrix('D:\蓝知医疗\嵌入式\youngs modulus calculation\youngs modulus.xls');
x = data(:,1);
y = data(:,2);

%%
    segments = [1, 2, 3, 13];
    fts = {'poly1', 'poly1', 'poly1'};

    [fit_y, cross_x] = piecewiseLineFunc(x,y,segments,fts);
    

    startPoints = {fit_y, segments, cross_x};
    ft = fittype( 'piecewiseLine(x,fit_y,segments,cross_x)' );
    func = fit(x,y,ft,'StartPoint',startPoints);

    figure;
    plot(func,x,y);