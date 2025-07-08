% test_sfrmat5
% Peter Burns, 27 Feb 2023
dir_path = 'E:\Code\Optical_aberration_estimation1\dataset\54854_bayer\crop\shot0.01000_read0.00050';
save_path = 'E:\Code\Optical_aberration_estimation1\dataset\54854_bayer\mat\shot0.01000_read0.00050';
% dir_path = 'E:\Code\Optical_aberration_estimation1\dataset\54854_rgb\crop\shot0_read0';
% save_path = 'E:\Code\Optical_aberration_estimation1\dataset\54854_rgb\mat\shot0_read0';

if exist(save_path, 'dir') == 0
    % 如果文件夹不存在，则创建该文件夹
    mkdir(folderPath);
end
% 获取文件夹中的所有文件
fileList = dir(fullfile(dir_path, '*.mat'));

io = 1;
del = 1;
npol = 5;
wflag = 0;
% 循环读取每个文件
for i = 1:numel(fileList)
    filename = fileList(i).name;
    filepath = fullfile(dir_path, filename);
    pattern = 'fov(-?\d+\.\d+)_angle(-?\d+\.\d+)';
    tokens = regexp(filename, pattern, 'tokens', 'once');

    % 提取浮点数
    fov = str2double(tokens{1});
    rot = str2double(tokens{2});
    
    % 读取tif图像
    mat = load(filepath);
    mat = mat.bayer;
    %image = imread(filepath);
    
    sfr_l = zeros(49,5);
    for j = 1:4      
        [status, sfr, e, sfr50, fitme, esf, nbin, del2] = sfrmat5(io, del, mat(:,:,j), npol);
        sfr_l(:,j+1) = sfr(:,2);
    end 
    sfr_l(:,1) = sfr(:,1);
    % 在这里可以对图像进行进一步处理
    if max(max(sfr_l(:,2)))==1 && ~any(isnan(sfr_l(:,2)))
        % 准备要保存的数据
        sfr = sfr_l;
        % 指定保存文件的路径和文件名
%         savefilename = strrep(filename, '.tif', '.mat');
        saveFile = fullfile(save_path,filename);

        if isnan(sum(sum(sfr)))
            disp('数组中存在 NaN 值，无法执行下面的语句。');
        else
            % 保存数据为.mat文件
            save(saveFile, 'fov','rot','sfr');
        end
    else
        filename
    end

end

