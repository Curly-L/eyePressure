%% 文件读取和输出模块，负责所有的文件读写工作
% 包括：读取文件夹下的特定格式文件的路径和文件名

% function F = eye_pressure
%     F.combine_string = @combine_string;
%     F.eye_pressure1 = @eye_pressure1;
%     F.getFilesPath = @getFilesPath;
%     F.readMeasureData = @readMeasureData;
%     F.getNoiseData = @getNoiseData;
%     F.getEyesData = @getEyesData;
%     F.getdataLR = @getdataLR;
%     F.strSort = @strSort;
%     F.getMuSig = @getMuSig;
%     F.gaussianFit = @gaussianFit;
%     F.gaussianMuSigma = @gaussianMuSigma;
%     F.selectRegion = @selectRegion;
%     F.generateGaussianData = @generateGaussianData;
%     F.getOutliers = @getOutliers;
%     F.gaussianPlotAll = @gaussianPlotAll;
%     F.truncStartEnds = @truncStartEnds;
%     F.correctData = @correctData;
%     F.fitData  = @fitData;
% end

function [combined_string] = combine_string(str1,str2_list,i)
    % 串联字符串和字符串元胞数组的值，并返回合并后的字符串
    % str1是要串联的字符串
    % str2_list是字符串元胞数组
    % i是需要串联的字符串元胞数组的序号
    combined_string = strcat(str1,str2_list(i));
end

function [file,filepath] = getFilesPath(baseDir, ext, excludeFiles, findSubfile)
    % 读取文件夹下的特定格式文件的路径，可选择排除某几个文件
    % baseDir是搜索的基本路径， 不会超过这个范围
    % ext是后缀名，这两个是必须要有的
    % excludeFiles是需要排除的文件
    % findSubfile指的是否进行子文件夹迭代搜索
    % [file,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles); 子文件夹迭代搜索
    % [file,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles, '\'); 仅当前文件夹搜索
    if nargin == 2 || isempty(excludeFiles)
        excludFileFlag = false;
    else
        excludFileFlag = true;
    end
    if nargin == 3
        findSubfile = true;
    else
        findSubfile = false;
    end

    if findSubfile
        dirOutput = dir([baseDir '/**/*.' ext]);
    else
        dirOutput = dir([baseDir '/*.', ext]);
    end

    if excludFileFlag
        dirOutput = dirOutput(~ismember({dirOutput.name}, excludeFiles));
    end

    folder = string({dirOutput.folder}');
    file = string({dirOutput.name}');
    filepath = strcat(folder, '\', file);
end

function  [num,txt,raw] = readMeasureData(filepath)
    [num,txt,raw] = xlsread(filepath);
end

function [msResult] = getNoiseData(file_dir)
    % 读取空跑文件数据
    % file_dir空跑文件路径
    [~,filepath] = getFilesPath(file_dir, 'xlsx');
    num_tests_noise = length(filepath);
    msResult = [];
    for i = 1:num_tests_noise
        fileread = filepath(i);
        [num,~,~] = readMeasureData(fileread);
        msResult = [msResult;num(:,2)]; % 1/2 高/低通滤波结果，暂时取2
    end
end

function [msResult] = getEyesData(file_dir)
    % 读取单个人的左眼/右眼测量数据，可以迭代搜索子文件夹
    % file_dir单人眼压数据文件路径
    excludeFiles = {'zcx测试情况统计表.xlsx'};
    [~,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles);
    [filepathL,filepathR] = getdataLR(filepath);
    filepathL = strSort(filepathL);
    filepathR = strSort(filepathR);
    num_tests = length(filepathL);
    num_eyes = 2;
    filepaths = [filepathL filepathR];
    msResult = cell(num_tests,num_eyes);
    for eye = 1:num_eyes
        % 1/2  左眼/右眼
        filepath = filepaths(:,eye);
        for i = 1:num_tests
            fileread = filepath(i);
            [num,~,~] = readMeasureData(fileread);
            msResult{i,eye} = num(:,2); % 1/2 高/低通滤波结果，暂时取2
        end
    end
end


%% 文件格式预处理模块
% 包括：文件索引预处理，文件内容预处理

% 文件索引预处理
function [dataL,dataR] = getdataLR(filepath)
    % 根据是否包含字符分类
    containsL = contains(filepath, '左眼');
    containsR = contains(filepath, '右眼');
    % 分类结果
    dataL = filepath(containsL);
    dataR = filepath(containsR);
end

function dataNew = strSort(data)
    % 以-作为分割的文件名，找到最后一个（.前）的数字
    splitData = split(data, '-');
    lastGroup = splitData(:,end);
    splitData = split(lastGroup, '.');
    lastGroup = splitData(:,1);
    % 排序
    lastGroupNum = str2double(lastGroup);
    [~, sortIdx] = sort(lastGroupNum);
    dataNew = data(sortIdx);
end

% 文件格式预处理
% 包括文件内容格式的预处理，此处暂时不需要

%% 数据分析预处理模块
% 包括求均值和标准差，高斯拟合，修正空跑误差，生成x距离坐标

% 计算数据参数
function [mu,sigma] = getMuSig(data)
    % 求均值和标准差
    mu = mean(data);
    sigma = std(data);
end

function [fitresult ,gof] = gaussianFit(x,y)
    % 求高斯拟合函数
    [fitresult ,gof] = fit(x, y,'gauss1');
end

% 阶段处理函数
function [datas] = truncStartEnds(data)
    % 截取接触到眼皮的压力曲线，其中包含了x坐标的生成，后期需要独立并且单独处理讨论（严格步进）
    function  [xNew,yNew] = truncStartEnd(num)
        row = length(num);
        x= (1:row)';
        x = x.*0.00567; % 严格步进的情况
        y = num;
        mask = y<0;
        fd1 = find(mask(1:600) == 1);
        fd2 = find(mask(600:length(mask)) == 1);
        if(~isempty(fd1))
            start_x =fd1(length(fd1))+1;
        else 
            start_x = 1;
        end
        
        if(~isempty(fd2))
            end_x =600+fd2(1)-1;
        else
            end_x = length(y);
        end
        xNew = x(start_x:end_x);
        yNew = y(start_x:end_x);
    end

    num_tests = length(data);

    datas = cell(num_tests,2);
    for i = 1:num_tests
        num = data{i};
        [xNew,yNew] = truncStartEnd(num);

        datas{i,1} = xNew;
        datas{i,2} = yNew;
    end
    
end

function [dataNew] = correctData(data,bias)
    % 主要修正空跑自身的系统误差
    % data左右眼数据
    % bias空跑数据平均值
    % 暂时的处理方案是数据-空跑数据平均值mu
    dataSize = size(data); %size返回[行 列]
    dataNew = cell(dataSize);
    num_tests = dataSize(1);
    for i = 1:num_tests
        dataNew{i,2} = data{i,2} - bias;
        dataNew{i,1} = data{i,1};
    end
end

function [dataNew] = fitData(data)
    % 对元胞数组中的数据进行高斯拟合
    dataSize = size(data);
    dataNew = cell(dataSize);
    num_tests = dataSize(1);
    for i = 1:num_tests
        x = data{i,1};
        y = data{i,2};
        [fitresult ,gof] =  gaussianFit(x,y);
        dataNew{i,1} = fitresult;
        dataNew{i,2} = gof;
    end
end

    % 计算单眼高斯曲线的平均值和不确定度
function [mu,sigma] =  gaussianMuSigma(fit_data)
    dataSize = size(fit_data);
    num_datas = dataSize(1);
    mu = mean(fit_data,1);
    stdfit = std(fit_data);
    sigma = sqrt(stdfit/(num_datas));
end

function [start_x, end_x] = selectRegion(data)
    % 选择讨论区间：所有测量x的最大和最小值
    all_data = data{:,1};
    start_x = min(all_data);
    end_x = max(all_data);
end

function [dataNew] = generateGaussianData(data,x)
    % 通过拟合的高斯参数和选择区间生成矩阵
    dataSize = size(data);
    num_tests = dataSize(1);
    dataNew = cell(num_tests,1);
    for i = 1:num_tests
        fitresult = data{i,1};
        yfit = feval(fitresult, x);
        dataNew{i} = yfit;
    end    
end

function num = getOutliers(data)
    % 算保留95%的中间数据时要去掉的单边的极值曲线数量
    dataSize = size(data);
    num_tests = dataSize(1);
    num = cell(num_tests*0.95)/2;
end

%% 绘图模块
% 高斯分布集中画在一张图
function [] = gaussianPlotAll(ori_data,fit_data,mode,originalDataFlag)
    % ori_data低通滤波的单眼横纵原数据
    % fit_data高斯拟合的参数值
    % mode模式选择，origin和select有什么区别？
    % originalDataFlag是否绘制原始数据
    if nargin ==3
        originalDataFlag = false;
    end
    figure;
    xlabel("Distance(mm)");
    ylabel("Force(mmHg)");
    dataSize = size(fit_data);
    num_tests = dataSize(1);

    for i = 1:num_tests
        hold on;
        fitresult = fit_data{i,1};
        if mode == "origin"
            x = ori_data{i,1};
            y = ori_data{i,2};
            originalDataFlag = true; %没看懂,mode和originalDataFlag功能重合
        elseif mode == "select"
            x = ori_data{i,1};
        end
        if originalDataFlag
            yfit = 5*feval(fitresult, x);
            plot(x,yfit,x,y);
        else
            yfit = 5*feval(fitresult, x);
            plot(x,yfit);
        end
        legend off;
    end
end

% % 画单眼的高斯曲线平均值曲线和+-3sigma的曲线
% function [] = meanGaussianPlot(ori_data,fit_data,mode)
% 
% end

function [fit_data_aligned] = GaussianFitAlign(fit_data)
%brief:把高斯cfit类型的一列cell,b1统一成，a1最大的那个cell的b1
%para:fit_data 第一列是高斯拟合的类型cell
%return:b1统一后的cfit类型的一列cell
    [row,~] = size(fit_data);
    gaussian_tempa1 = zeros(row,1);
    for i=1:row
    gaussian_tempa1(i) = fit_data{i}.a1;
    end
    [~,max_ind] = max(gaussian_tempa1);
    for i=1:row
    fit_data{i}.b1 = fit_data{max_ind}.b1;
    end
    fit_data_aligned = fit_data;
end

function [xdots] = CrossDots(fit_data,y)
%brief:给定高斯参数，和y,求高斯函数等于y时的根，（交点）
%para:fit_data 给定cfit类型的一个cell
%para：y y值
%return:xdots 交点的x值
a1 = fit_data.a1;
b1 = fit_data.b1;
c1 = fit_data.c1;
Fun = @(x)(a1.*exp(-((x-b1)/c1).^2))-y;
dots(1) = fzero(Fun,0.5);
dots(2) = fzero(Fun,4);
xdots = dots;
end

function [xdots,x,y,index] = CrossDots_batch(fit_data)
%brief:给定一列高斯cfit类型，求非最高曲线的最高点的y值，和最高曲线的交点,最高曲线的最高点y值和自己的交点，就是他的最高点
%para:fit_data 给定cfit类型的一列cell
%return:xdots 一列cell,每个cell是交点的两个横坐标（横坐标是1x2数组）
%return:x 最高曲线顶点坐标x
%return:y 最高曲线顶点坐标y
    %找到a1的最大值
    [row,~] = size(fit_data);
    [a1,b1,~] = getGaussionCoeff(fit_data);
    [max_val,max_ind1] = max(a1);
    y = max_val;
    x = b1(max_ind1);
    index = max_ind1;
    %用小于a1的最大值的a1求交点,求一列
    temp = cell(row,1);
    for i = 1:row
        if(i == max_ind1)
            temp{i} = [x,x];
        else
            temp{i} = CrossDots(fit_data{max_ind1,1},a1(i));
        end
    end
    xdots =temp;
end

function [coeff_a1,coeff_b1,coeff_c1] = getGaussionCoeff(fit_data)
%brief:获取一列高斯cfit类型的a1,b1,c1三个参数
%para:fit_data 给定cfit类型的一列cell
%para：
%return:a1参数列
%return:b1参数列
%return:c1参数列
    [row,~] = size(fit_data);
    coeff_a1 = zeros(row,1);
    coeff_b1 = zeros(row,1);
    coeff_c1 = zeros(row,1);
    for i=1:row
        coeff_a1(i) = fit_data{i}.a1;
        coeff_b1(i) = fit_data{i}.b1;
        coeff_c1(i) = fit_data{i}.c1;
    end
end

function [bias] = getBias(x_list,value)
%brief:获得矮曲线，从中间分开后，应该左右移动的距离
%para:x_list 一列cell，每个cell [x1,x2]
%para：value 最高曲线中轴线的x
%return:bias 一列cell,每个cell,[x1,x2]应该向左，和向右移动的距离，是正数
    [row,~] = size(x_list);
    bias = cell(row,1);
    for i=1:row
        bias_left = value - x_list{i,1}(1);
        bias_right = x_list{i,1}(2) - value;
        bias{i,1} = [bias_left,bias_right];
    end
end

function [] = plotAll(x_list,y_list)
%brief:画图,画
%para:x_list 一列cell，每个cell 是一列x坐标数组
%para：y_list 一列cell，每个cell 是一列y坐标数组
%return:
    figure;
    xlabel("Distance(mm)");
    ylabel("Force(mmHg)");
    
    [row,~] = size(x_list);
    for i=1:row
        hold on
        plot(x_list{i,1},y_list{i,1});
        legend off
    end
end

function [data,index] = dataInsert(list,value)
%brief:往list中插入一个值，插入到list中第一个比value大的数的前面
%para:list 一列数组
%para:value 插入的值
%return:data 插入数后的数组
%return:index 插入的位置
    shape = size(list);
    for i=1:shape(1)
        if(list(i)>value)
            data = [list(1:i-1)',value,list(i:shape(1))']';
            index = i;
            break;
        end
    end
end

function [data,index] = dataInsert_batchx(list,value)
%brief:往list中批量插入一个值，插入到list中第一个比value大的数的前面
%para:list 一列cell,每个cell是一列数组
%para:value 插入的值
%return:data 插入数后的一列cell
%return:index 插入的位置(数组)
    shape = size(list);
    data = cell(shape(1),1);
    index = zeros(shape(1),1);
    data_temp = zeros(shape(1),1);
    for j=1:shape(1)
        [data_temp,index_temp] = dataInsert(list{j,1},value);
        data{j} = data_temp;
        index(j) = index_temp;
    end
end

function [data] = dataInsert_Index(list,value,ind)
%brief:往list中插入一个值，插入到list(ind)这里
%para:list 一列数组
%para：value 插入的值
%para：ind 插入的值
%return:data 插入数后的数组
    shape = size(list);    
    data = [list(1:ind-1)',value,list(ind:shape(1))']';
end


function [data] = dataInsert_batchy(list,value,index)
%brief:往list中批量插入一个值，插入到指定位置
%para:list 一列cell,每个cell是一列数组
%para：value 插入的值
%para：index 插入的位置（数组）
%return:data 插入数后的一列cell
%note:后期value可能也改成数组形式，现在是一个值
    shape = size(list);
    data = cell(shape(1),1);
    for j=1:shape(1)
        data{j,1} = dataInsert_Index(list{j,1},value,index(j));
    end
end

function  data = getCoeff(list)
%brief获得一个数组的最大，最小，平均，标准差
%para:list 一列数组
%return:懒得写了
    [max_val,~] = max(list); 
    [min_val,~] = min(list);
    mu = mean(list);
    sigma = std(list);
    data = [max_val,min_val,mu,sigma];
end

function data_new = newCell2Mat(x,yfit_new)
% 将单眼cell数组转换成[num_tests,num_datas]的double数组
    testSize = size(yfit_new);
    dataSize = size(x);
    num_tests = testSize(1);
    num_datas = dataSize(2);
    data_new = zeros(num_tests,num_datas);
    for i = 1:num_tests
        data_new(i,:) = feval(yfit_new{i,1},x);
    end
end

    % 算单眼与icare标定的比例系数
    % mean_data是处理后的单眼高斯拟合的平均值曲线
    % stdData是用其他眼压测量设备测得的标定值
    % k是缩放均值曲线的比例系数，使缩放后的峰值和标定值相等
function k = calibrateProportion(mean_data,stdData)
    gaussionPeak = 5*max(mean_data);
    k = stdData/gaussionPeak;
end

%% 主函数，程序入口

% 工作路径
file_dir_noise = 'D:\蓝知医疗\测试数据\空跑';
msResult_noise = getNoiseData(file_dir_noise);
[mu,sigma] = getMuSig(msResult_noise); % 计算空跑平均值和标准差

file_dir = 'D:\蓝知医疗\测试数据\kz(icare测不出来眼睛睁不开)_次数命名';
excludeFiles = {'情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\laL22R23';
% excludeFiles = {'0803李昂情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\lffL15R12';
% excludeFiles = {'测试情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\lpjL16R13';
% excludeFiles = {'0731-0801测试数据说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\zcxL17R17';
% excludeFiles = {'zcx测试情况统计表.xlsx'};
msResult = getEyesData(file_dir); % 得到左右眼眼压，行-测量次数，列-被测眼睛
[num_tests,num_eyes] = size(msResult);

% 初始设置
gaussianPlotAllFlag = true;
meanGaussianPlotFlag = true;

% 函数调用和处理

% 截断处理（利用空跑数据确定可用数据范围）
% 去除小于零的无效数据
msResultTrunc = cell(num_tests,2);
msResultTruncs = cell(2,1);
for k = 1:num_eyes  % 左右眼分开处理
    msResultk = msResult(:,k);
    msResultTrunc  = truncStartEnds(msResultk);
    msResultTruncs{k} = msResultTrunc;
end

% 空跑数据校正
msResultCorrect = cell(num_tests,2);
msResultCorrects = cell(2,1);
for k = 1:num_eyes % 左右眼分开处理
    msResultCorrect  = correctData(msResultTruncs{k},mu);
    msResultCorrects{k} = msResultCorrect;
end

% 高斯数据拟合 
% 得到的是拟合参数
msResultFit = cell(num_tests,2);
msResultFits = cell(2,1);
for k = 1:num_eyes % 左右眼分开处理
    msResultFit  = fitData(msResultCorrects{k});
    msResultFits{k} = msResultFit;
end

%高斯拟合之后对齐(b1一样)
alligned_fit = cell(2,1);
alligned_fit{1,1} = GaussianFitAlign(msResultFits{1});
alligned_fit{2,1} = GaussianFitAlign(msResultFits{2});
%求非最高曲线的最高点的y值，和最高曲线的交点
cross_x = cell(2,1);
[cross_x{1,1},x1,y1,ind1] = CrossDots_batch(alligned_fit{1,1});
[cross_x{2,1},x2,y2,ind2] = CrossDots_batch(alligned_fit{2,1});
ymax = [y1,y2];
xmax = [x1,x2];
%求非最高曲线左右移动的偏差
bias = cell(2,1);
bias{1,1} = getBias(cross_x{1,1},x1);
bias{2,1} = getBias(cross_x{2,1},x2);
%x移动
%移动步骤之一，创建对齐后的xy
x_move = 0.4:0.005:6.8;
x_move = x_move';
[row,~] = size(alligned_fit{1,1});
yfit = cell(row,1);
yfit1 = cell(2,1);
xfit = cell(row,1);
xfit1 = cell(2,1);
for i= 1:num_eyes %yfit和xfit作为临时存储，储存单次测量拟合后的数据
    for j=1:row
         yfit{j,1} = feval(alligned_fit{i,1}{j,1}, x_move);
         xfit{j,1} = x_move;
    end
    yfit1{i,1} = yfit;
    xfit1{i,1} = xfit;
end
xfit2 = xfit1;%xfit2拿去画图用，移动之前的x值拿去画图
%移动步骤之二，移动x
len = length(xfit1{1,1}{i,1});
for k = 1:num_eyes
    for j= 1:row
        for i = 1:len
            if (xfit1{k,1}{j,1}(i)<alligned_fit{k,1}{1,1}.b1)
                xfit1{k,1}{j,1}(i) = xfit1{k,1}{j,1}(i) - bias{k,1}{j,1}(1);
            else
                xfit1{k,1}{j,1}(i) = xfit1{k,1}{j,1}(i) + bias{k,1}{j,1}(2);
            end
        end   
    end
end

%移动完了之后，加一个最高点
len1 = length(xfit1{1,1});
insert_index = cell(2,1);
for j = 1:num_eyes
        [xfit1{j,1},insert_index{j}] = dataInsert_batchx(xfit1{j,1},xmax(j));
        yfit1{j,1} = dataInsert_batchy(yfit1{j,1},ymax(j),insert_index{j});
end

% 高斯数据重新拟合
fitData_new = cell(num_tests,2);
fitData_new1 = cell(2,1);
msResultFit_new = cell(num_tests,2);
msResultFits_new = cell(2,1);
for j =1:num_eyes
    for i=1:num_tests
        fitData_new{i,1} = xfit1{j,1}{i,1};
        fitData_new{i,2} = yfit1{j,1}{i,1};
    end
    fitData_new1{j,1} = fitData_new;
end

% for k = 1:num_eyes
%     msResultFit_new  = fitData(fitData_new1{k});
%     msResultFits_new{k} = msResultFit_new;
% end

% 
fitt = fittype('a1*exp(-(x-b1)^2/(2*c1^2))');
options = optimoptions('lsqcurvefit', 'TolFun', 1e-6, 'TolX', 1e-6, 'MaxIter', 1000); 
%TolFun（函数容差）和 TolX（步长容差）等参数，以控制优化的精度

for j = 1:num_eyes
    a1 = ymax(j);
    b1 = xmax(j);
    x0 = [6 0.5];
    func = @(c1,x) a1*exp(-(x-b1).^2./(2*c1.^2));
    for i=1:num_tests
        c1 = lsqcurvefit(func,x0(j),xfit1{j}{i},fitData_new1{j}{i,2}, [], [], options);
        msResultFits_new{j}{i,1} = cfit(fitt, a1, b1, c1);
    end
end

[a1,b1,~] = getGaussionCoeff(msResultFits_new{1,1});
[a2,b2,~] = getGaussionCoeff(msResultFits_new{2,1});

coeff = cell(2,1);
coeff_temp = zeros(4,1);
coeff_temp = getCoeff(a1)';
coeff{1,1} = coeff_temp;
coeff_temp = getCoeff(a2)';
coeff{2,1} = coeff_temp;

%% 画Gaussian拟合的图
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(msResultCorrects{k},msResultFits{k},mode);
    end
end

% 画对齐后的图
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(msResultCorrects{k},alligned_fit{k},mode);
    end
end

if gaussianPlotAllFlag
    for k = 1:num_eyes
        plotAll(xfit1{k},yfit1{k});
    end
end

% 画新拟合的图
% 横坐标与
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(xfit2{k},msResultFits_new{k},mode);
    end
end

%% cell转mat，计算均值和不确定度
xfit_new = cell(2,1);
mu_new = cell(2,1);
sigma_new = cell(2,1);
% xfit_new{1} = msResultCorrects{1}{ind1,1};
% xfit_new{2} = msResultCorrects{2}{ind2,1};
xfit_new{1} = 0:0.005:10;
xfit_new{2} = 0:0.005:10;
data_new = cell(2,1);
p = zeros(2,1);
stdData = [17,17];

if gaussianPlotAllFlag
    for k = 1:num_eyes
        data_new{k} = newCell2Mat(xfit_new{k},msResultFits_new{k});
        [mu_new{k},sigma_new{k}] =  gaussianMuSigma(data_new{k});
        p = calibrateProportion(mu_new{k},10);
        figure;
        plot(xfit_new{k},5*p*mu_new{k},xfit_new{k},5*p*(mu_new{k}+sigma_new{k}),xfit_new{k},5*p*(mu_new{k}-sigma_new{k}));
        yline(stdData(k));
        xlabel("Distance(mm)");
        ylabel("Force(mmHg)");
        if k==1
            title("left eye");
        else 
            title("right eye");
        end
    end
end

% %  Gaussian拟合的参数作为标准数据进行处理
% 
% % 曲线均值和不确定度
% % gaussianMus = zeros(2,1);
% % gaussianSigmas = zeros(2,1);
% % msResultFitDatas = cell(2,1);
% % for k = 1:num_eyes % 左右眼分开处理，用高斯拟合的参数算曲线
% %     
% %     msResultFitDatas{k} = generateGaussianData(msResultFits{k},xFitDatas);
% % end
% % if meanGaussianPlotFlag
% %     for k = 1:num_eyes % 左右眼分开处理
% %         [gaussianMus(k),gaussianSigmas(k)] = gaussianMuSigma(msResultFitDatas{k});
% %     end
% % end
% 
% 
% % % 选定区间
% % start_xs = cell(num_eyes,1);
% % end_xs = cell(num_eyes,1);
% % xs = cell(num_eyes,1);
% % for k = 1:num_eyes % 左右眼分开处理
% %     [start_xs{k}, end_xs{k}] = selectRegion(msResultCorrects{k}); % 选择曲线讨论区间
% %     xs{k} = linspace(start_xs{k},end_xs{k})';
% % end
% 
% % % 选定区间规范化数据（高斯数据拟合公式在对应的x区间内的矩阵）
% % for k = 1:num_eyes % 左右眼分开处理
% %     msResultCorrects{k} = generateGaussianData(msResultFits{k},xs{k}); % 选择曲线讨论区间
% %     % 注意，这里替换了msResultCorrects的内容
% % end
% 
% 
% % % 画Gaussian拟合的图
% % for k = 1:num_eyes
% %     mode = "select";
% %     gaussianPlotAll(xs{k},msResultFits{k},mode);
% % end
% 
% %end


%% 文件读取和输出模块，负责所有的文件读写工作
% 包括：读取文件夹下的特定格式文件的路径和文件名

% function F = eye_pressure
%     F.combine_string = @combine_string;
%     F.eye_pressure1 = @eye_pressure1;
%     F.getFilesPath = @getFilesPath;
%     F.readMeasureData = @readMeasureData;
%     F.getNoiseData = @getNoiseData;
%     F.getEyesData = @getEyesData;
%     F.getdataLR = @getdataLR;
%     F.strSort = @strSort;
%     F.getMuSig = @getMuSig;
%     F.gaussianFit = @gaussianFit;
%     F.gaussianMuSigma = @gaussianMuSigma;
%     F.selectRegion = @selectRegion;
%     F.generateGaussianData = @generateGaussianData;
%     F.getOutliers = @getOutliers;
%     F.gaussianPlotAll = @gaussianPlotAll;
%     F.truncStartEnds = @truncStartEnds;
%     F.correctData = @correctData;
%     F.fitData  = @fitData;
% end

function [combined_string] = combine_string(str1,str2_list,i)
    % 串联字符串和字符串元胞数组的值，并返回合并后的字符串
    % str1是要串联的字符串
    % str2_list是字符串元胞数组
    % i是需要串联的字符串元胞数组的序号
    combined_string = strcat(str1,str2_list(i));
end

function [file,filepath] = getFilesPath(baseDir, ext, excludeFiles, findSubfile)
    % 读取文件夹下的特定格式文件的路径，可选择排除某几个文件
    % baseDir是搜索的基本路径， 不会超过这个范围
    % ext是后缀名，这两个是必须要有的
    % excludeFiles是需要排除的文件
    % findSubfile指的是否进行子文件夹迭代搜索
    % [file,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles); 子文件夹迭代搜索
    % [file,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles, '\'); 仅当前文件夹搜索
    if nargin == 2 || isempty(excludeFiles)
        excludFileFlag = false;
    else
        excludFileFlag = true;
    end
    if nargin == 3
        findSubfile = true;
    else
        findSubfile = false;
    end

    if findSubfile
        dirOutput = dir([baseDir '/**/*.' ext]);
    else
        dirOutput = dir([baseDir '/*.', ext]);
    end

    if excludFileFlag
        dirOutput = dirOutput(~ismember({dirOutput.name}, excludeFiles));
    end

    folder = string({dirOutput.folder}');
    file = string({dirOutput.name}');
    filepath = strcat(folder, '\', file);
end

function  [num,txt,raw] = readMeasureData(filepath)
    [num,txt,raw] = xlsread(filepath);
end

function [msResult] = getNoiseData(file_dir)
    % 读取空跑文件数据
    % file_dir空跑文件路径
    [~,filepath] = getFilesPath(file_dir, 'xlsx');
    num_tests_noise = length(filepath);
    msResult = [];
    for i = 1:num_tests_noise
        fileread = filepath(i);
        [num,~,~] = readMeasureData(fileread);
        msResult = [msResult;num(:,2)]; % 1/2 高/低通滤波结果，暂时取2
    end
end

function [msResult] = getEyesData(file_dir)
    % 读取单个人的左眼/右眼测量数据，可以迭代搜索子文件夹
    % file_dir单人眼压数据文件路径
    excludeFiles = {'zcx测试情况统计表.xlsx'};
    [~,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles);
    [filepathL,filepathR] = getdataLR(filepath);
    filepathL = strSort(filepathL);
    filepathR = strSort(filepathR);
    num_tests = length(filepathL);
    num_eyes = 2;
    filepaths = [filepathL filepathR];
    msResult = cell(num_tests,num_eyes);
    for eye = 1:num_eyes
        % 1/2  左眼/右眼
        filepath = filepaths(:,eye);
        for i = 1:num_tests
            fileread = filepath(i);
            [num,~,~] = readMeasureData(fileread);
            msResult{i,eye} = num(:,2); % 1/2 高/低通滤波结果，暂时取2
        end
    end
end


%% 文件格式预处理模块
% 包括：文件索引预处理，文件内容预处理

% 文件索引预处理
function [dataL,dataR] = getdataLR(filepath)
    % 根据是否包含字符分类
    containsL = contains(filepath, '左眼');
    containsR = contains(filepath, '右眼');
    % 分类结果
    dataL = filepath(containsL);
    dataR = filepath(containsR);
end

function dataNew = strSort(data)
    % 以-作为分割的文件名，找到最后一个（.前）的数字
    splitData = split(data, '-');
    lastGroup = splitData(:,end);
    splitData = split(lastGroup, '.');
    lastGroup = splitData(:,1);
    % 排序
    lastGroupNum = str2double(lastGroup);
    [~, sortIdx] = sort(lastGroupNum);
    dataNew = data(sortIdx);
end

% 文件格式预处理
% 包括文件内容格式的预处理，此处暂时不需要

%% 数据分析预处理模块
% 包括求均值和标准差，高斯拟合，修正空跑误差，生成x距离坐标

% 计算数据参数
function [mu,sigma] = getMuSig(data)
    % 求均值和标准差
    mu = mean(data);
    sigma = std(data);
end

function [fitresult ,gof] = gaussianFit(x,y)
    % 求高斯拟合函数
    [fitresult ,gof] = fit(x, y,'gauss1');
end

% 阶段处理函数
function [datas] = truncStartEnds(data)
    % 截取接触到眼皮的压力曲线，其中包含了x坐标的生成，后期需要独立并且单独处理讨论（严格步进）
    function  [xNew,yNew] = truncStartEnd(num)
        row = length(num);
        x= (1:row)';
        x = x.*0.00567; % 严格步进的情况
        y = num;
        mask = y<0;
        fd1 = find(mask(1:600) == 1);
        fd2 = find(mask(600:length(mask)) == 1);
        if(~isempty(fd1))
            start_x =fd1(length(fd1))+1;
        else 
            start_x = 1;
        end
        
        if(~isempty(fd2))
            end_x =600+fd2(1)-1;
        else
            end_x = length(y);
        end
        xNew = x(start_x:end_x);
        yNew = y(start_x:end_x);
    end

    num_tests = length(data);

    datas = cell(num_tests,2);
    for i = 1:num_tests
        num = data{i};
        [xNew,yNew] = truncStartEnd(num);

        datas{i,1} = xNew;
        datas{i,2} = yNew;
    end
    
end

function [dataNew] = correctData(data,bias)
    % 主要修正空跑自身的系统误差
    % data左右眼数据
    % bias空跑数据平均值
    % 暂时的处理方案是数据-空跑数据平均值mu
    dataSize = size(data); %size返回[行 列]
    dataNew = cell(dataSize);
    num_tests = dataSize(1);
    for i = 1:num_tests
        dataNew{i,2} = data{i,2} - bias;
        dataNew{i,1} = data{i,1};
    end
end

function [dataNew] = fitData(data)
    % 对元胞数组中的数据进行高斯拟合
    dataSize = size(data);
    dataNew = cell(dataSize);
    num_tests = dataSize(1);
    for i = 1:num_tests
        x = data{i,1};
        y = data{i,2};
        [fitresult ,gof] =  gaussianFit(x,y);
        dataNew{i,1} = fitresult;
        dataNew{i,2} = gof;
    end
end

    % 计算单眼高斯曲线的平均值和不确定度
function [mu,sigma] =  gaussianMuSigma(fit_data)
    dataSize = size(fit_data);
    num_datas = dataSize(1);
    mu = mean(fit_data,1);
    stdfit = std(fit_data);
    sigma = sqrt(stdfit/(num_datas));
end

function [start_x, end_x] = selectRegion(data)
    % 选择讨论区间：所有测量x的最大和最小值
    all_data = data{:,1};
    start_x = min(all_data);
    end_x = max(all_data);
end

function [dataNew] = generateGaussianData(data,x)
    % 通过拟合的高斯参数和选择区间生成矩阵
    dataSize = size(data);
    num_tests = dataSize(1);
    dataNew = cell(num_tests,1);
    for i = 1:num_tests
        fitresult = data{i,1};
        yfit = feval(fitresult, x);
        dataNew{i} = yfit;
    end    
end

function num = getOutliers(data)
    % 算保留95%的中间数据时要去掉的单边的极值曲线数量
    dataSize = size(data);
    num_tests = dataSize(1);
    num = cell(num_tests*0.95)/2;
end

%% 绘图模块
% 高斯分布集中画在一张图
function [] = gaussianPlotAll(ori_data,fit_data,mode,originalDataFlag)
    % ori_data低通滤波的单眼横纵原数据
    % fit_data高斯拟合的参数值
    % mode模式选择，origin和select有什么区别？
    % originalDataFlag是否绘制原始数据
    if nargin ==3
        originalDataFlag = false;
    end
    figure;
    xlabel("Distance(mm)");
    ylabel("Force(mmHg)");
    dataSize = size(fit_data);
    num_tests = dataSize(1);

    for i = 1:num_tests
        hold on;
        fitresult = fit_data{i,1};
        if mode == "origin"
            x = ori_data{i,1};
            y = ori_data{i,2};
            originalDataFlag = true; %没看懂,mode和originalDataFlag功能重合
        elseif mode == "select"
            x = ori_data{i,1};
        end
        if originalDataFlag
            yfit = 5*feval(fitresult, x);
            plot(x,yfit,x,y);
        else
            yfit = 5*feval(fitresult, x);
            plot(x,yfit);
        end
        legend off;
    end
end

% % 画单眼的高斯曲线平均值曲线和+-3sigma的曲线
% function [] = meanGaussianPlot(ori_data,fit_data,mode)
% 
% end

function [fit_data_aligned] = GaussianFitAlign(fit_data)
%brief:把高斯cfit类型的一列cell,b1统一成，a1最大的那个cell的b1
%para:fit_data 第一列是高斯拟合的类型cell
%return:b1统一后的cfit类型的一列cell
    [row,~] = size(fit_data);
    gaussian_tempa1 = zeros(row,1);
    for i=1:row
    gaussian_tempa1(i) = fit_data{i}.a1;
    end
    [~,max_ind] = max(gaussian_tempa1);
    for i=1:row
    fit_data{i}.b1 = fit_data{max_ind}.b1;
    end
    fit_data_aligned = fit_data;
end

function [xdots] = CrossDots(fit_data,y)
%brief:给定高斯参数，和y,求高斯函数等于y时的根，（交点）
%para:fit_data 给定cfit类型的一个cell
%para：y y值
%return:xdots 交点的x值
a1 = fit_data.a1;
b1 = fit_data.b1;
c1 = fit_data.c1;
Fun = @(x)(a1.*exp(-((x-b1)/c1).^2))-y;
dots(1) = fzero(Fun,0.5);
dots(2) = fzero(Fun,4);
xdots = dots;
end

function [xdots,x,y,index] = CrossDots_batch(fit_data)
%brief:给定一列高斯cfit类型，求非最高曲线的最高点的y值，和最高曲线的交点,最高曲线的最高点y值和自己的交点，就是他的最高点
%para:fit_data 给定cfit类型的一列cell
%return:xdots 一列cell,每个cell是交点的两个横坐标（横坐标是1x2数组）
%return:x 最高曲线顶点坐标x
%return:y 最高曲线顶点坐标y
    %找到a1的最大值
    [row,~] = size(fit_data);
    [a1,b1,~] = getGaussionCoeff(fit_data);
    [max_val,max_ind1] = max(a1);
    y = max_val;
    x = b1(max_ind1);
    index = max_ind1;
    %用小于a1的最大值的a1求交点,求一列
    temp = cell(row,1);
    for i = 1:row
        if(i == max_ind1)
            temp{i} = [x,x];
        else
            temp{i} = CrossDots(fit_data{max_ind1,1},a1(i));
        end
    end
    xdots =temp;
end

function [coeff_a1,coeff_b1,coeff_c1] = getGaussionCoeff(fit_data)
%brief:获取一列高斯cfit类型的a1,b1,c1三个参数
%para:fit_data 给定cfit类型的一列cell
%para：
%return:a1参数列
%return:b1参数列
%return:c1参数列
    [row,~] = size(fit_data);
    coeff_a1 = zeros(row,1);
    coeff_b1 = zeros(row,1);
    coeff_c1 = zeros(row,1);
    for i=1:row
        coeff_a1(i) = fit_data{i}.a1;
        coeff_b1(i) = fit_data{i}.b1;
        coeff_c1(i) = fit_data{i}.c1;
    end
end

function [bias] = getBias(x_list,value)
%brief:获得矮曲线，从中间分开后，应该左右移动的距离
%para:x_list 一列cell，每个cell [x1,x2]
%para：value 最高曲线中轴线的x
%return:bias 一列cell,每个cell,[x1,x2]应该向左，和向右移动的距离，是正数
    [row,~] = size(x_list);
    bias = cell(row,1);
    for i=1:row
        bias_left = value - x_list{i,1}(1);
        bias_right = x_list{i,1}(2) - value;
        bias{i,1} = [bias_left,bias_right];
    end
end

function [] = plotAll(x_list,y_list)
%brief:画图,画
%para:x_list 一列cell，每个cell 是一列x坐标数组
%para：y_list 一列cell，每个cell 是一列y坐标数组
%return:
    figure;
    xlabel("Distance(mm)");
    ylabel("Force(mmHg)");
    
    [row,~] = size(x_list);
    for i=1:row
        hold on
        plot(x_list{i,1},y_list{i,1});
        legend off
    end
end

function [data,index] = dataInsert(list,value)
%brief:往list中插入一个值，插入到list中第一个比value大的数的前面
%para:list 一列数组
%para:value 插入的值
%return:data 插入数后的数组
%return:index 插入的位置
    shape = size(list);
    for i=1:shape(1)
        if(list(i)>value)
            data = [list(1:i-1)',value,list(i:shape(1))']';
            index = i;
            break;
        end
    end
end

function [data,index] = dataInsert_batchx(list,value)
%brief:往list中批量插入一个值，插入到list中第一个比value大的数的前面
%para:list 一列cell,每个cell是一列数组
%para:value 插入的值
%return:data 插入数后的一列cell
%return:index 插入的位置(数组)
    shape = size(list);
    data = cell(shape(1),1);
    index = zeros(shape(1),1);
    data_temp = zeros(shape(1),1);
    for j=1:shape(1)
        [data_temp,index_temp] = dataInsert(list{j,1},value);
        data{j} = data_temp;
        index(j) = index_temp;
    end
end

function [data] = dataInsert_Index(list,value,ind)
%brief:往list中插入一个值，插入到list(ind)这里
%para:list 一列数组
%para：value 插入的值
%para：ind 插入的值
%return:data 插入数后的数组
    shape = size(list);    
    data = [list(1:ind-1)',value,list(ind:shape(1))']';
end


function [data] = dataInsert_batchy(list,value,index)
%brief:往list中批量插入一个值，插入到指定位置
%para:list 一列cell,每个cell是一列数组
%para：value 插入的值
%para：index 插入的位置（数组）
%return:data 插入数后的一列cell
%note:后期value可能也改成数组形式，现在是一个值
    shape = size(list);
    data = cell(shape(1),1);
    for j=1:shape(1)
        data{j,1} = dataInsert_Index(list{j,1},value,index(j));
    end
end

function  data = getCoeff(list)
%brief获得一个数组的最大，最小，平均，标准差
%para:list 一列数组
%return:懒得写了
    [max_val,~] = max(list); 
    [min_val,~] = min(list);
    mu = mean(list);
    sigma = std(list);
    data = [max_val,min_val,mu,sigma];
end

function data_new = newCell2Mat(x,yfit_new)
% 将单眼cell数组转换成[num_tests,num_datas]的double数组
    testSize = size(yfit_new);
    dataSize = size(x);
    num_tests = testSize(1);
    num_datas = dataSize(2);
    data_new = zeros(num_tests,num_datas);
    for i = 1:num_tests
        data_new(i,:) = feval(yfit_new{i,1},x);
    end
end

    % 算单眼与icare标定的比例系数
    % mean_data是处理后的单眼高斯拟合的平均值曲线
    % stdData是用其他眼压测量设备测得的标定值
    % k是缩放均值曲线的比例系数，使缩放后的峰值和标定值相等
function k = calibrateProportion(mean_data,stdData)
    gaussionPeak = 5*max(mean_data);
    k = stdData/gaussionPeak;
end

%% 主函数，程序入口

% 工作路径
file_dir_noise = 'D:\蓝知医疗\测试数据\空跑';
msResult_noise = getNoiseData(file_dir_noise);
[mu,sigma] = getMuSig(msResult_noise); % 计算空跑平均值和标准差

file_dir = 'D:\蓝知医疗\测试数据\kz(icare测不出来眼睛睁不开)_次数命名';
excludeFiles = {'情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\laL22R23';
% excludeFiles = {'0803李昂情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\lffL15R12';
% excludeFiles = {'测试情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\lpjL16R13';
% excludeFiles = {'0731-0801测试数据说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\zcxL17R17';
% excludeFiles = {'zcx测试情况统计表.xlsx'};
msResult = getEyesData(file_dir); % 得到左右眼眼压，行-测量次数，列-被测眼睛
[num_tests,num_eyes] = size(msResult);

% 初始设置
gaussianPlotAllFlag = true;
meanGaussianPlotFlag = true;

% 函数调用和处理

% 截断处理（利用空跑数据确定可用数据范围）
% 去除小于零的无效数据
msResultTrunc = cell(num_tests,2);
msResultTruncs = cell(2,1);
for k = 1:num_eyes  % 左右眼分开处理
    msResultk = msResult(:,k);
    msResultTrunc  = truncStartEnds(msResultk);
    msResultTruncs{k} = msResultTrunc;
end

% 空跑数据校正
msResultCorrect = cell(num_tests,2);
msResultCorrects = cell(2,1);
for k = 1:num_eyes % 左右眼分开处理
    msResultCorrect  = correctData(msResultTruncs{k},mu);
    msResultCorrects{k} = msResultCorrect;
end

% 高斯数据拟合 
% 得到的是拟合参数
msResultFit = cell(num_tests,2);
msResultFits = cell(2,1);
for k = 1:num_eyes % 左右眼分开处理
    msResultFit  = fitData(msResultCorrects{k});
    msResultFits{k} = msResultFit;
end

%高斯拟合之后对齐(b1一样)
alligned_fit = cell(2,1);
alligned_fit{1,1} = GaussianFitAlign(msResultFits{1});
alligned_fit{2,1} = GaussianFitAlign(msResultFits{2});
%求非最高曲线的最高点的y值，和最高曲线的交点
cross_x = cell(2,1);
[cross_x{1,1},x1,y1,ind1] = CrossDots_batch(alligned_fit{1,1});
[cross_x{2,1},x2,y2,ind2] = CrossDots_batch(alligned_fit{2,1});
ymax = [y1,y2];
xmax = [x1,x2];
%求非最高曲线左右移动的偏差
bias = cell(2,1);
bias{1,1} = getBias(cross_x{1,1},x1);
bias{2,1} = getBias(cross_x{2,1},x2);
%x移动
%移动步骤之一，创建对齐后的xy
x_move = 0.4:0.005:6.8;
x_move = x_move';
[row,~] = size(alligned_fit{1,1});
yfit = cell(row,1);
yfit1 = cell(2,1);
xfit = cell(row,1);
xfit1 = cell(2,1);
for i= 1:num_eyes %yfit和xfit作为临时存储，储存单次测量拟合后的数据
    for j=1:row
         yfit{j,1} = feval(alligned_fit{i,1}{j,1}, x_move);
         xfit{j,1} = x_move;
    end
    yfit1{i,1} = yfit;
    xfit1{i,1} = xfit;
end
xfit2 = xfit1;%xfit2拿去画图用，移动之前的x值拿去画图
%移动步骤之二，移动x
len = length(xfit1{1,1}{i,1});
for k = 1:num_eyes
    for j= 1:row
        for i = 1:len
            if (xfit1{k,1}{j,1}(i)<alligned_fit{k,1}{1,1}.b1)
                xfit1{k,1}{j,1}(i) = xfit1{k,1}{j,1}(i) - bias{k,1}{j,1}(1);
            else
                xfit1{k,1}{j,1}(i) = xfit1{k,1}{j,1}(i) + bias{k,1}{j,1}(2);
            end
        end   
    end
end

%移动完了之后，加一个最高点
len1 = length(xfit1{1,1});
insert_index = cell(2,1);
for j = 1:num_eyes
        [xfit1{j,1},insert_index{j}] = dataInsert_batchx(xfit1{j,1},xmax(j));
        yfit1{j,1} = dataInsert_batchy(yfit1{j,1},ymax(j),insert_index{j});
end

% 高斯数据重新拟合
fitData_new = cell(num_tests,2);
fitData_new1 = cell(2,1);
msResultFit_new = cell(num_tests,2);
msResultFits_new = cell(2,1);
for j =1:num_eyes
    for i=1:num_tests
        fitData_new{i,1} = xfit1{j,1}{i,1};
        fitData_new{i,2} = yfit1{j,1}{i,1};
    end
    fitData_new1{j,1} = fitData_new;
end

% for k = 1:num_eyes
%     msResultFit_new  = fitData(fitData_new1{k});
%     msResultFits_new{k} = msResultFit_new;
% end

% 
fitt = fittype('a1*exp(-(x-b1)^2/(2*c1^2))');
options = optimoptions('lsqcurvefit', 'TolFun', 1e-6, 'TolX', 1e-6, 'MaxIter', 1000); 
%TolFun（函数容差）和 TolX（步长容差）等参数，以控制优化的精度

for j = 1:num_eyes
    a1 = ymax(j);
    b1 = xmax(j);
    x0 = [6 0.5];
    func = @(c1,x) a1*exp(-(x-b1).^2./(2*c1.^2));
    for i=1:num_tests
        c1 = lsqcurvefit(func,x0(j),xfit1{j}{i},fitData_new1{j}{i,2}, [], [], options);
        msResultFits_new{j}{i,1} = cfit(fitt, a1, b1, c1);
    end
end

[a1,b1,~] = getGaussionCoeff(msResultFits_new{1,1});
[a2,b2,~] = getGaussionCoeff(msResultFits_new{2,1});

coeff = cell(2,1);
coeff_temp = zeros(4,1);
coeff_temp = getCoeff(a1)';
coeff{1,1} = coeff_temp;
coeff_temp = getCoeff(a2)';
coeff{2,1} = coeff_temp;

%% 画Gaussian拟合的图
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(msResultCorrects{k},msResultFits{k},mode);
    end
end

% 画对齐后的图
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(msResultCorrects{k},alligned_fit{k},mode);
    end
end

if gaussianPlotAllFlag
    for k = 1:num_eyes
        plotAll(xfit1{k},yfit1{k});
    end
end

% 画新拟合的图
% 横坐标与
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(xfit2{k},msResultFits_new{k},mode);
    end
end

%% cell转mat，计算均值和不确定度
xfit_new = cell(2,1);
mu_new = cell(2,1);
sigma_new = cell(2,1);
% xfit_new{1} = msResultCorrects{1}{ind1,1};
% xfit_new{2} = msResultCorrects{2}{ind2,1};
xfit_new{1} = 0:0.005:10;
xfit_new{2} = 0:0.005:10;
data_new = cell(2,1);
p = zeros(2,1);
stdData = [17,17];

if gaussianPlotAllFlag
    for k = 1:num_eyes
        data_new{k} = newCell2Mat(xfit_new{k},msResultFits_new{k});
        [mu_new{k},sigma_new{k}] =  gaussianMuSigma(data_new{k});
        p = calibrateProportion(mu_new{k},10);
        figure;
        plot(xfit_new{k},5*p*mu_new{k},xfit_new{k},5*p*(mu_new{k}+sigma_new{k}),xfit_new{k},5*p*(mu_new{k}-sigma_new{k}));
        yline(stdData(k));
        xlabel("Distance(mm)");
        ylabel("Force(mmHg)");
        if k==1
            title("left eye");
        else 
            title("right eye");
        end
    end
end

% %  Gaussian拟合的参数作为标准数据进行处理
% 
% % 曲线均值和不确定度
% % gaussianMus = zeros(2,1);
% % gaussianSigmas = zeros(2,1);
% % msResultFitDatas = cell(2,1);
% % for k = 1:num_eyes % 左右眼分开处理，用高斯拟合的参数算曲线
% %     
% %     msResultFitDatas{k} = generateGaussianData(msResultFits{k},xFitDatas);
% % end
% % if meanGaussianPlotFlag
% %     for k = 1:num_eyes % 左右眼分开处理
% %         [gaussianMus(k),gaussianSigmas(k)] = gaussianMuSigma(msResultFitDatas{k});
% %     end
% % end
% 
% 
% % % 选定区间
% % start_xs = cell(num_eyes,1);
% % end_xs = cell(num_eyes,1);
% % xs = cell(num_eyes,1);
% % for k = 1:num_eyes % 左右眼分开处理
% %     [start_xs{k}, end_xs{k}] = selectRegion(msResultCorrects{k}); % 选择曲线讨论区间
% %     xs{k} = linspace(start_xs{k},end_xs{k})';
% % end
% 
% % % 选定区间规范化数据（高斯数据拟合公式在对应的x区间内的矩阵）
% % for k = 1:num_eyes % 左右眼分开处理
% %     msResultCorrects{k} = generateGaussianData(msResultFits{k},xs{k}); % 选择曲线讨论区间
% %     % 注意，这里替换了msResultCorrects的内容
% % end
% 
% 
% % % 画Gaussian拟合的图
% % for k = 1:num_eyes
% %     mode = "select";
% %     gaussianPlotAll(xs{k},msResultFits{k},mode);
% % end
% 
% %end


%% 文件读取和输出模块，负责所有的文件读写工作
% 包括：读取文件夹下的特定格式文件的路径和文件名

% function F = eye_pressure
%     F.combine_string = @combine_string;
%     F.eye_pressure1 = @eye_pressure1;
%     F.getFilesPath = @getFilesPath;
%     F.readMeasureData = @readMeasureData;
%     F.getNoiseData = @getNoiseData;
%     F.getEyesData = @getEyesData;
%     F.getdataLR = @getdataLR;
%     F.strSort = @strSort;
%     F.getMuSig = @getMuSig;
%     F.gaussianFit = @gaussianFit;
%     F.gaussianMuSigma = @gaussianMuSigma;
%     F.selectRegion = @selectRegion;
%     F.generateGaussianData = @generateGaussianData;
%     F.getOutliers = @getOutliers;
%     F.gaussianPlotAll = @gaussianPlotAll;
%     F.truncStartEnds = @truncStartEnds;
%     F.correctData = @correctData;
%     F.fitData  = @fitData;
% end

function [combined_string] = combine_string(str1,str2_list,i)
    % 串联字符串和字符串元胞数组的值，并返回合并后的字符串
    % str1是要串联的字符串
    % str2_list是字符串元胞数组
    % i是需要串联的字符串元胞数组的序号
    combined_string = strcat(str1,str2_list(i));
end

function [file,filepath] = getFilesPath(baseDir, ext, excludeFiles, findSubfile)
    % 读取文件夹下的特定格式文件的路径，可选择排除某几个文件
    % baseDir是搜索的基本路径， 不会超过这个范围
    % ext是后缀名，这两个是必须要有的
    % excludeFiles是需要排除的文件
    % findSubfile指的是否进行子文件夹迭代搜索
    % [file,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles); 子文件夹迭代搜索
    % [file,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles, '\'); 仅当前文件夹搜索
    if nargin == 2 || isempty(excludeFiles)
        excludFileFlag = false;
    else
        excludFileFlag = true;
    end
    if nargin == 3
        findSubfile = true;
    else
        findSubfile = false;
    end

    if findSubfile
        dirOutput = dir([baseDir '/**/*.' ext]);
    else
        dirOutput = dir([baseDir '/*.', ext]);
    end

    if excludFileFlag
        dirOutput = dirOutput(~ismember({dirOutput.name}, excludeFiles));
    end

    folder = string({dirOutput.folder}');
    file = string({dirOutput.name}');
    filepath = strcat(folder, '\', file);
end

function  [num,txt,raw] = readMeasureData(filepath)
    [num,txt,raw] = xlsread(filepath);
end

function [msResult] = getNoiseData(file_dir)
    % 读取空跑文件数据
    % file_dir空跑文件路径
    [~,filepath] = getFilesPath(file_dir, 'xlsx');
    num_tests_noise = length(filepath);
    msResult = [];
    for i = 1:num_tests_noise
        fileread = filepath(i);
        [num,~,~] = readMeasureData(fileread);
        msResult = [msResult;num(:,2)]; % 1/2 高/低通滤波结果，暂时取2
    end
end

function [msResult] = getEyesData(file_dir)
    % 读取单个人的左眼/右眼测量数据，可以迭代搜索子文件夹
    % file_dir单人眼压数据文件路径
    excludeFiles = {'zcx测试情况统计表.xlsx'};
    [~,filepath] = getFilesPath(file_dir, 'xlsx', excludeFiles);
    [filepathL,filepathR] = getdataLR(filepath);
    filepathL = strSort(filepathL);
    filepathR = strSort(filepathR);
    num_tests = length(filepathL);
    num_eyes = 2;
    filepaths = [filepathL filepathR];
    msResult = cell(num_tests,num_eyes);
    for eye = 1:num_eyes
        % 1/2  左眼/右眼
        filepath = filepaths(:,eye);
        for i = 1:num_tests
            fileread = filepath(i);
            [num,~,~] = readMeasureData(fileread);
            msResult{i,eye} = num(:,2); % 1/2 高/低通滤波结果，暂时取2
        end
    end
end


%% 文件格式预处理模块
% 包括：文件索引预处理，文件内容预处理

% 文件索引预处理
function [dataL,dataR] = getdataLR(filepath)
    % 根据是否包含字符分类
    containsL = contains(filepath, '左眼');
    containsR = contains(filepath, '右眼');
    % 分类结果
    dataL = filepath(containsL);
    dataR = filepath(containsR);
end

function dataNew = strSort(data)
    % 以-作为分割的文件名，找到最后一个（.前）的数字
    splitData = split(data, '-');
    lastGroup = splitData(:,end);
    splitData = split(lastGroup, '.');
    lastGroup = splitData(:,1);
    % 排序
    lastGroupNum = str2double(lastGroup);
    [~, sortIdx] = sort(lastGroupNum);
    dataNew = data(sortIdx);
end

% 文件格式预处理
% 包括文件内容格式的预处理，此处暂时不需要

%% 数据分析预处理模块
% 包括求均值和标准差，高斯拟合，修正空跑误差，生成x距离坐标

% 计算数据参数
function [mu,sigma] = getMuSig(data)
    % 求均值和标准差
    mu = mean(data);
    sigma = std(data);
end

function [fitresult ,gof] = gaussianFit(x,y)
    % 求高斯拟合函数
    [fitresult ,gof] = fit(x, y,'gauss1');
end

% 阶段处理函数
function [datas] = truncStartEnds(data)
    % 截取接触到眼皮的压力曲线，其中包含了x坐标的生成，后期需要独立并且单独处理讨论（严格步进）
    function  [xNew,yNew] = truncStartEnd(num)
        row = length(num);
        x= (1:row)';
        x = x.*0.00567; % 严格步进的情况
        y = num;
        mask = y<0;
        fd1 = find(mask(1:600) == 1);
        fd2 = find(mask(600:length(mask)) == 1);
        if(~isempty(fd1))
            start_x =fd1(length(fd1))+1;
        else 
            start_x = 1;
        end
        
        if(~isempty(fd2))
            end_x =600+fd2(1)-1;
        else
            end_x = length(y);
        end
        xNew = x(start_x:end_x);
        yNew = y(start_x:end_x);
    end

    num_tests = length(data);

    datas = cell(num_tests,2);
    for i = 1:num_tests
        num = data{i};
        [xNew,yNew] = truncStartEnd(num);

        datas{i,1} = xNew;
        datas{i,2} = yNew;
    end
    
end

function [dataNew] = correctData(data,bias)
    % 主要修正空跑自身的系统误差
    % data左右眼数据
    % bias空跑数据平均值
    % 暂时的处理方案是数据-空跑数据平均值mu
    dataSize = size(data); %size返回[行 列]
    dataNew = cell(dataSize);
    num_tests = dataSize(1);
    for i = 1:num_tests
        dataNew{i,2} = data{i,2} - bias;
        dataNew{i,1} = data{i,1};
    end
end

function [dataNew] = fitData(data)
    % 对元胞数组中的数据进行高斯拟合
    dataSize = size(data);
    dataNew = cell(dataSize);
    num_tests = dataSize(1);
    for i = 1:num_tests
        x = data{i,1};
        y = data{i,2};
        [fitresult ,gof] =  gaussianFit(x,y);
        dataNew{i,1} = fitresult;
        dataNew{i,2} = gof;
    end
end

    % 计算单眼高斯曲线的平均值和不确定度
function [mu,sigma] =  gaussianMuSigma(fit_data)
    dataSize = size(fit_data);
    num_datas = dataSize(1);
    mu = mean(fit_data,1);
    stdfit = std(fit_data);
    sigma = sqrt(stdfit/(num_datas));
end

function [start_x, end_x] = selectRegion(data)
    % 选择讨论区间：所有测量x的最大和最小值
    all_data = data{:,1};
    start_x = min(all_data);
    end_x = max(all_data);
end

function [dataNew] = generateGaussianData(data,x)
    % 通过拟合的高斯参数和选择区间生成矩阵
    dataSize = size(data);
    num_tests = dataSize(1);
    dataNew = cell(num_tests,1);
    for i = 1:num_tests
        fitresult = data{i,1};
        yfit = feval(fitresult, x);
        dataNew{i} = yfit;
    end    
end

function num = getOutliers(data)
    % 算保留95%的中间数据时要去掉的单边的极值曲线数量
    dataSize = size(data);
    num_tests = dataSize(1);
    num = cell(num_tests*0.95)/2;
end

%% 绘图模块
% 高斯分布集中画在一张图
function [] = gaussianPlotAll(ori_data,fit_data,mode,originalDataFlag)
    % ori_data低通滤波的单眼横纵原数据
    % fit_data高斯拟合的参数值
    % mode模式选择，origin和select有什么区别？
    % originalDataFlag是否绘制原始数据
    if nargin ==3
        originalDataFlag = false;
    end
    figure;
    xlabel("Distance(mm)");
    ylabel("Force(mmHg)");
    dataSize = size(fit_data);
    num_tests = dataSize(1);

    for i = 1:num_tests
        hold on;
        fitresult = fit_data{i,1};
        if mode == "origin"
            x = ori_data{i,1};
            y = ori_data{i,2};
            originalDataFlag = true; %没看懂,mode和originalDataFlag功能重合
        elseif mode == "select"
            x = ori_data{i,1};
        end
        if originalDataFlag
            yfit = 5*feval(fitresult, x);
            plot(x,yfit,x,y);
        else
            yfit = 5*feval(fitresult, x);
            plot(x,yfit);
        end
        legend off;
    end
end

% % 画单眼的高斯曲线平均值曲线和+-3sigma的曲线
% function [] = meanGaussianPlot(ori_data,fit_data,mode)
% 
% end

function [fit_data_aligned] = GaussianFitAlign(fit_data)
%brief:把高斯cfit类型的一列cell,b1统一成，a1最大的那个cell的b1
%para:fit_data 第一列是高斯拟合的类型cell
%return:b1统一后的cfit类型的一列cell
    [row,~] = size(fit_data);
    gaussian_tempa1 = zeros(row,1);
    for i=1:row
    gaussian_tempa1(i) = fit_data{i}.a1;
    end
    [~,max_ind] = max(gaussian_tempa1);
    for i=1:row
    fit_data{i}.b1 = fit_data{max_ind}.b1;
    end
    fit_data_aligned = fit_data;
end

function [xdots] = CrossDots(fit_data,y)
%brief:给定高斯参数，和y,求高斯函数等于y时的根，（交点）
%para:fit_data 给定cfit类型的一个cell
%para：y y值
%return:xdots 交点的x值
a1 = fit_data.a1;
b1 = fit_data.b1;
c1 = fit_data.c1;
Fun = @(x)(a1.*exp(-((x-b1)/c1).^2))-y;
dots(1) = fzero(Fun,0.5);
dots(2) = fzero(Fun,4);
xdots = dots;
end

function [xdots,x,y,index] = CrossDots_batch(fit_data)
%brief:给定一列高斯cfit类型，求非最高曲线的最高点的y值，和最高曲线的交点,最高曲线的最高点y值和自己的交点，就是他的最高点
%para:fit_data 给定cfit类型的一列cell
%return:xdots 一列cell,每个cell是交点的两个横坐标（横坐标是1x2数组）
%return:x 最高曲线顶点坐标x
%return:y 最高曲线顶点坐标y
    %找到a1的最大值
    [row,~] = size(fit_data);
    [a1,b1,~] = getGaussionCoeff(fit_data);
    [max_val,max_ind1] = max(a1);
    y = max_val;
    x = b1(max_ind1);
    index = max_ind1;
    %用小于a1的最大值的a1求交点,求一列
    temp = cell(row,1);
    for i = 1:row
        if(i == max_ind1)
            temp{i} = [x,x];
        else
            temp{i} = CrossDots(fit_data{max_ind1,1},a1(i));
        end
    end
    xdots =temp;
end

function [coeff_a1,coeff_b1,coeff_c1] = getGaussionCoeff(fit_data)
%brief:获取一列高斯cfit类型的a1,b1,c1三个参数
%para:fit_data 给定cfit类型的一列cell
%para：
%return:a1参数列
%return:b1参数列
%return:c1参数列
    [row,~] = size(fit_data);
    coeff_a1 = zeros(row,1);
    coeff_b1 = zeros(row,1);
    coeff_c1 = zeros(row,1);
    for i=1:row
        coeff_a1(i) = fit_data{i}.a1;
        coeff_b1(i) = fit_data{i}.b1;
        coeff_c1(i) = fit_data{i}.c1;
    end
end

function [bias] = getBias(x_list,value)
%brief:获得矮曲线，从中间分开后，应该左右移动的距离
%para:x_list 一列cell，每个cell [x1,x2]
%para：value 最高曲线中轴线的x
%return:bias 一列cell,每个cell,[x1,x2]应该向左，和向右移动的距离，是正数
    [row,~] = size(x_list);
    bias = cell(row,1);
    for i=1:row
        bias_left = value - x_list{i,1}(1);
        bias_right = x_list{i,1}(2) - value;
        bias{i,1} = [bias_left,bias_right];
    end
end

function [] = plotAll(x_list,y_list)
%brief:画图,画
%para:x_list 一列cell，每个cell 是一列x坐标数组
%para：y_list 一列cell，每个cell 是一列y坐标数组
%return:
    figure;
    xlabel("Distance(mm)");
    ylabel("Force(mmHg)");
    
    [row,~] = size(x_list);
    for i=1:row
        hold on
        plot(x_list{i,1},y_list{i,1});
        legend off
    end
end

function [data,index] = dataInsert(list,value)
%brief:往list中插入一个值，插入到list中第一个比value大的数的前面
%para:list 一列数组
%para:value 插入的值
%return:data 插入数后的数组
%return:index 插入的位置
    shape = size(list);
    for i=1:shape(1)
        if(list(i)>value)
            data = [list(1:i-1)',value,list(i:shape(1))']';
            index = i;
            break;
        end
    end
end

function [data,index] = dataInsert_batchx(list,value)
%brief:往list中批量插入一个值，插入到list中第一个比value大的数的前面
%para:list 一列cell,每个cell是一列数组
%para:value 插入的值
%return:data 插入数后的一列cell
%return:index 插入的位置(数组)
    shape = size(list);
    data = cell(shape(1),1);
    index = zeros(shape(1),1);
    data_temp = zeros(shape(1),1);
    for j=1:shape(1)
        [data_temp,index_temp] = dataInsert(list{j,1},value);
        data{j} = data_temp;
        index(j) = index_temp;
    end
end

function [data] = dataInsert_Index(list,value,ind)
%brief:往list中插入一个值，插入到list(ind)这里
%para:list 一列数组
%para：value 插入的值
%para：ind 插入的值
%return:data 插入数后的数组
    shape = size(list);    
    data = [list(1:ind-1)',value,list(ind:shape(1))']';
end


function [data] = dataInsert_batchy(list,value,index)
%brief:往list中批量插入一个值，插入到指定位置
%para:list 一列cell,每个cell是一列数组
%para：value 插入的值
%para：index 插入的位置（数组）
%return:data 插入数后的一列cell
%note:后期value可能也改成数组形式，现在是一个值
    shape = size(list);
    data = cell(shape(1),1);
    for j=1:shape(1)
        data{j,1} = dataInsert_Index(list{j,1},value,index(j));
    end
end

function  data = getCoeff(list)
%brief获得一个数组的最大，最小，平均，标准差
%para:list 一列数组
%return:懒得写了
    [max_val,~] = max(list); 
    [min_val,~] = min(list);
    mu = mean(list);
    sigma = std(list);
    data = [max_val,min_val,mu,sigma];
end

function data_new = newCell2Mat(x,yfit_new)
% 将单眼cell数组转换成[num_tests,num_datas]的double数组
    testSize = size(yfit_new);
    dataSize = size(x);
    num_tests = testSize(1);
    num_datas = dataSize(2);
    data_new = zeros(num_tests,num_datas);
    for i = 1:num_tests
        data_new(i,:) = feval(yfit_new{i,1},x);
    end
end

    % 算单眼与icare标定的比例系数
    % mean_data是处理后的单眼高斯拟合的平均值曲线
    % stdData是用其他眼压测量设备测得的标定值
    % k是缩放均值曲线的比例系数，使缩放后的峰值和标定值相等
function k = calibrateProportion(mean_data,stdData)
    gaussionPeak = 5*max(mean_data);
    k = stdData/gaussionPeak;
end

%% 主函数，程序入口

% 工作路径
file_dir_noise = 'D:\蓝知医疗\测试数据\空跑';
msResult_noise = getNoiseData(file_dir_noise);
[mu,sigma] = getMuSig(msResult_noise); % 计算空跑平均值和标准差

file_dir = 'D:\蓝知医疗\测试数据\kz(icare测不出来眼睛睁不开)_次数命名';
excludeFiles = {'情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\laL22R23';
% excludeFiles = {'0803李昂情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\lffL15R12';
% excludeFiles = {'测试情况说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\lpjL16R13';
% excludeFiles = {'0731-0801测试数据说明.xlsx'};
% file_dir = 'D:\蓝知医疗\测试数据\zcxL17R17';
% excludeFiles = {'zcx测试情况统计表.xlsx'};
msResult = getEyesData(file_dir); % 得到左右眼眼压，行-测量次数，列-被测眼睛
[num_tests,num_eyes] = size(msResult);

% 初始设置
gaussianPlotAllFlag = true;
meanGaussianPlotFlag = true;

% 函数调用和处理

% 截断处理（利用空跑数据确定可用数据范围）
% 去除小于零的无效数据
msResultTrunc = cell(num_tests,2);
msResultTruncs = cell(2,1);
for k = 1:num_eyes  % 左右眼分开处理
    msResultk = msResult(:,k);
    msResultTrunc  = truncStartEnds(msResultk);
    msResultTruncs{k} = msResultTrunc;
end

% 空跑数据校正
msResultCorrect = cell(num_tests,2);
msResultCorrects = cell(2,1);
for k = 1:num_eyes % 左右眼分开处理
    msResultCorrect  = correctData(msResultTruncs{k},mu);
    msResultCorrects{k} = msResultCorrect;
end

% 高斯数据拟合 
% 得到的是拟合参数
msResultFit = cell(num_tests,2);
msResultFits = cell(2,1);
for k = 1:num_eyes % 左右眼分开处理
    msResultFit  = fitData(msResultCorrects{k});
    msResultFits{k} = msResultFit;
end

%高斯拟合之后对齐(b1一样)
alligned_fit = cell(2,1);
alligned_fit{1,1} = GaussianFitAlign(msResultFits{1});
alligned_fit{2,1} = GaussianFitAlign(msResultFits{2});
%求非最高曲线的最高点的y值，和最高曲线的交点
cross_x = cell(2,1);
[cross_x{1,1},x1,y1,ind1] = CrossDots_batch(alligned_fit{1,1});
[cross_x{2,1},x2,y2,ind2] = CrossDots_batch(alligned_fit{2,1});
ymax = [y1,y2];
xmax = [x1,x2];
%求非最高曲线左右移动的偏差
bias = cell(2,1);
bias{1,1} = getBias(cross_x{1,1},x1);
bias{2,1} = getBias(cross_x{2,1},x2);
%x移动
%移动步骤之一，创建对齐后的xy
x_move = 0.4:0.005:6.8;
x_move = x_move';
[row,~] = size(alligned_fit{1,1});
yfit = cell(row,1);
yfit1 = cell(2,1);
xfit = cell(row,1);
xfit1 = cell(2,1);
for i= 1:num_eyes %yfit和xfit作为临时存储，储存单次测量拟合后的数据
    for j=1:row
         yfit{j,1} = feval(alligned_fit{i,1}{j,1}, x_move);
         xfit{j,1} = x_move;
    end
    yfit1{i,1} = yfit;
    xfit1{i,1} = xfit;
end
xfit2 = xfit1;%xfit2拿去画图用，移动之前的x值拿去画图
%移动步骤之二，移动x
len = length(xfit1{1,1}{i,1});
for k = 1:num_eyes
    for j= 1:row
        for i = 1:len
            if (xfit1{k,1}{j,1}(i)<alligned_fit{k,1}{1,1}.b1)
                xfit1{k,1}{j,1}(i) = xfit1{k,1}{j,1}(i) - bias{k,1}{j,1}(1);
            else
                xfit1{k,1}{j,1}(i) = xfit1{k,1}{j,1}(i) + bias{k,1}{j,1}(2);
            end
        end   
    end
end

%移动完了之后，加一个最高点
len1 = length(xfit1{1,1});
insert_index = cell(2,1);
for j = 1:num_eyes
        [xfit1{j,1},insert_index{j}] = dataInsert_batchx(xfit1{j,1},xmax(j));
        yfit1{j,1} = dataInsert_batchy(yfit1{j,1},ymax(j),insert_index{j});
end

% 高斯数据重新拟合
fitData_new = cell(num_tests,2);
fitData_new1 = cell(2,1);
msResultFit_new = cell(num_tests,2);
msResultFits_new = cell(2,1);
for j =1:num_eyes
    for i=1:num_tests
        fitData_new{i,1} = xfit1{j,1}{i,1};
        fitData_new{i,2} = yfit1{j,1}{i,1};
    end
    fitData_new1{j,1} = fitData_new;
end

% for k = 1:num_eyes
%     msResultFit_new  = fitData(fitData_new1{k});
%     msResultFits_new{k} = msResultFit_new;
% end

% 
fitt = fittype('a1*exp(-(x-b1)^2/(2*c1^2))');
options = optimoptions('lsqcurvefit', 'TolFun', 1e-6, 'TolX', 1e-6, 'MaxIter', 1000); 
%TolFun（函数容差）和 TolX（步长容差）等参数，以控制优化的精度

for j = 1:num_eyes
    a1 = ymax(j);
    b1 = xmax(j);
    x0 = [6 0.5];
    func = @(c1,x) a1*exp(-(x-b1).^2./(2*c1.^2));
    for i=1:num_tests
        c1 = lsqcurvefit(func,x0(j),xfit1{j}{i},fitData_new1{j}{i,2}, [], [], options);
        msResultFits_new{j}{i,1} = cfit(fitt, a1, b1, c1);
    end
end

[a1,b1,~] = getGaussionCoeff(msResultFits_new{1,1});
[a2,b2,~] = getGaussionCoeff(msResultFits_new{2,1});

coeff = cell(2,1);
coeff_temp = zeros(4,1);
coeff_temp = getCoeff(a1)';
coeff{1,1} = coeff_temp;
coeff_temp = getCoeff(a2)';
coeff{2,1} = coeff_temp;

%% 画Gaussian拟合的图
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(msResultCorrects{k},msResultFits{k},mode);
    end
end

% 画对齐后的图
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(msResultCorrects{k},alligned_fit{k},mode);
    end
end

if gaussianPlotAllFlag
    for k = 1:num_eyes
        plotAll(xfit1{k},yfit1{k});
    end
end

% 画新拟合的图
% 横坐标与
if gaussianPlotAllFlag
    for k = 1:num_eyes
        mode = "select";
        gaussianPlotAll(xfit2{k},msResultFits_new{k},mode);
    end
end

%% cell转mat，计算均值和不确定度
xfit_new = cell(2,1);
mu_new = cell(2,1);
sigma_new = cell(2,1);
% xfit_new{1} = msResultCorrects{1}{ind1,1};
% xfit_new{2} = msResultCorrects{2}{ind2,1};
xfit_new{1} = 0:0.005:10;
xfit_new{2} = 0:0.005:10;
data_new = cell(2,1);
p = zeros(2,1);
stdData = [17,17];

if gaussianPlotAllFlag
    for k = 1:num_eyes
        data_new{k} = newCell2Mat(xfit_new{k},msResultFits_new{k});
        [mu_new{k},sigma_new{k}] =  gaussianMuSigma(data_new{k});
        p = calibrateProportion(mu_new{k},10);
        figure;
        plot(xfit_new{k},5*p*mu_new{k},xfit_new{k},5*p*(mu_new{k}+sigma_new{k}),xfit_new{k},5*p*(mu_new{k}-sigma_new{k}));
        yline(stdData(k));
        xlabel("Distance(mm)");
        ylabel("Force(mmHg)");
        if k==1
            title("left eye");
        else 
            title("right eye");
        end
    end
end

% %  Gaussian拟合的参数作为标准数据进行处理
% 
% % 曲线均值和不确定度
% % gaussianMus = zeros(2,1);
% % gaussianSigmas = zeros(2,1);
% % msResultFitDatas = cell(2,1);
% % for k = 1:num_eyes % 左右眼分开处理，用高斯拟合的参数算曲线
% %     
% %     msResultFitDatas{k} = generateGaussianData(msResultFits{k},xFitDatas);
% % end
% % if meanGaussianPlotFlag
% %     for k = 1:num_eyes % 左右眼分开处理
% %         [gaussianMus(k),gaussianSigmas(k)] = gaussianMuSigma(msResultFitDatas{k});
% %     end
% % end
% 
% 
% % % 选定区间
% % start_xs = cell(num_eyes,1);
% % end_xs = cell(num_eyes,1);
% % xs = cell(num_eyes,1);
% % for k = 1:num_eyes % 左右眼分开处理
% %     [start_xs{k}, end_xs{k}] = selectRegion(msResultCorrects{k}); % 选择曲线讨论区间
% %     xs{k} = linspace(start_xs{k},end_xs{k})';
% % end
% 
% % % 选定区间规范化数据（高斯数据拟合公式在对应的x区间内的矩阵）
% % for k = 1:num_eyes % 左右眼分开处理
% %     msResultCorrects{k} = generateGaussianData(msResultFits{k},xs{k}); % 选择曲线讨论区间
% %     % 注意，这里替换了msResultCorrects的内容
% % end
% 
% 
% % % 画Gaussian拟合的图
% % for k = 1:num_eyes
% %     mode = "select";
% %     gaussianPlotAll(xs{k},msResultFits{k},mode);
% % end
% 
% %end


