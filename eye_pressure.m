%% 文件读取和输出模块，负责所有的文件读写工作
% 包括：读取文件夹下的特定格式文件的路径和文件名

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

function [msResult] = getEyesData(file_dir,excludeFiles)
    % 读取单个人的左眼/右眼测量数据，可以迭代搜索子文件夹
    % file_dir单人眼压数据文件路径
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

    % 截取接触到眼皮的压力曲线，其中包含了x坐标的生成，后期需要独立并且单独处理讨论（严格步进）
function [datas] = truncStartEnds(data)
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


    % 主要修正空跑自身的系统误差
    % data左右眼数据
    % bias空跑数据平均值
    % 暂时的处理方案是数据-空跑数据平均值mu
function [dataNew] = correctData(data,bias)
    dataSize = size(data); %size返回[行 列]
    dataNew = cell(dataSize);
    num_tests = dataSize(1);
    for i = 1:num_tests
        dataNew{i,2} = data{i,2} - bias;
        dataNew{i,1} = data{i,1};
    end
end

    % 对元胞数组中的数据进行高斯拟合
function [dataNew] = fitData(data)
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
    dataSize = size(fit_data{1});
    num_datas = dataSize(1);
    mu = mean(fit_data,3);
    stdfit = std(fit_data);
    sigma = sqrt(stdfit/(num_datas));
end

    % 选择讨论区间：所有测量x的最大和最小值
function [start_x, end_x] = selectRegion(data)
    all_data = data{:,1};
    start_x = min(all_data);
    end_x = max(all_data);
end

    % 通过拟合的高斯参数和选择区间生成矩阵
function [dataNew] = generateGaussianData(data,x)
    dataSize = size(data);
    num_tests = dataSize(1);
    dataNew = cell(num_tests,1);
    for i = 1:num_tests
        fitresult = data{i,1};
        yfit = feval(fitresult, x);
        dataNew{i} = yfit;
    end
end

    % 算保留95%的中间数据时要去掉的单边的极值曲线数量
function num = getOutliers(data)
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
            y = 5*ori_data{i,2};
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

function [] = meanGaussianPlot(ori_data,fit_data,mode)

end

%% 主函数，程序入口

% 工作路径
file_dir_noise = 'D:\蓝知医疗\测试数据\空跑';
msResult_noise = getNoiseData(file_dir_noise);
[mu,sigma] = getMuSig(msResult_noise); % 计算空跑平均值和标准差

file_dir = 'D:\蓝知医疗\测试数据\lffL15R12';
excludeFiles = {'测试情况说明.xlsx'};
msResult = getEyesData(file_dir,excludeFiles); % 得到左右眼眼压，行-测量次数，列-被测眼睛
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

% 画Gaussian拟合的图
if gaussianPlotAllFlag
    for k = 1:num_eyes % 是否显示原始数据绘图
        mode = "select";
        gaussianPlotAll(msResultCorrects{k},msResultFits{k},mode);
    end
end

% Gaussian拟合的参数作为标准数据进行处理

% % 曲线均值和不确定度
% gaussianMus = zeros(2,1);
% gaussianSigmas = zeros(2,1);
% msResultFitDatas = cell(2,1);
% for k = 1:num_eyes % 左右眼分开处理，用高斯拟合的参数算曲线
% 
%     msResultFitDatas{k} = generateGaussianData(msResultFits{k},xFitDatas);
% end
% if meanGaussianPlotFlag
%     for k = 1:num_eyes % 左右眼分开处理
%         [gaussianMus(k),gaussianSigmas(k)] = gaussianMuSigma(msResultFitDatas{k});
%     end
% end

% 选定区间
% start_xs = cell(num_eyes,1);
% end_xs = cell(num_eyes,1);
% xs = cell(num_eyes,1);
% for k = 1:num_eyes % 左右眼分开处理
%     [start_xs{k}, end_xs{k}] = selectRegion(msResultCorrects{k}); % 选择曲线讨论区间
%     xs{k} = linspace(start_xs{k},end_xs{k})';
% end
% 
% % 选定区间规范化数据（高斯数据拟合公式在对应的x区间内的矩阵）
% for k = 1:num_eyes % 左右眼分开处理
%     msResultCorrects{k} = generateGaussianData(msResultFits{k},xs{k}); % 选择曲线讨论区间
%     % 注意，这里替换了msResultCorrects的内容
% end
% 
% % 画Gaussian拟合的图
% for k = 1:num_eyes
%     mode = "select";
%     gaussianPlotAll(xs{k},msResultFits{k},mode);
% end
