% This is the script for SEED_V EEG dataset preprocess, including:
% (1) Downsample
% (2) Band pass Filter
% (3) Divide into trials, specifically, data matrix for 1 subject 1 vedio
% (4) Detect and Interpolate bad channels
% (5) Auto ICA, remove artifacts, and line/eye/muscle noise components, based on matlab package EEGLab and Discover-eeg
% (6) Rereference
% (7) Reorder trials
% (8) Save data to .mat files

clear;
close all;
clc;

bpfreq = [0.5, 47];
debug = 0;
Fs = 200;

data_dir = "C:\Emotion_Data\SEED\Preprocessed_EEG";
% 获取所有符合条件的.mat文件
all_files = dir(fullfile(data_dir, '*_*.mat'));

% 提取所有被试名称
sub_names = cell(1, numel(all_files));
for f = 1:numel(all_files)
    [~, filename] = fileparts(all_files(f).name);
    split_name = strsplit(filename, '_');
    sub_names{f} = split_name{1}; % 提取下划线前的被试名称
end
unique_names = unique(sub_names);
n_sub = numel(unique_names);

channel_dir = 'C:\Emotion_Data\SEED'; % 请替换为实际路径
channel_file = 'chn_names.mat'; 

% 加载通道标签文件
channel_path = fullfile(channel_dir, channel_file);
if exist(channel_path, 'file')
    chan_data = load(channel_path);
    if isfield(chan_data, 'chn_names')
        hdr_labels = chan_data.chn_names;
        % 验证通道标签格式
        if ~iscell(hdr_labels) || ~all(cellfun(@ischar, hdr_labels))
            error('通道标签格式错误：chn_names应为包含字符串的cell数组');
        end
    else
        error('通道文件%s中未找到chn_names变量', channel_file);
    end
else
    error('通道文件不存在：%s', channel_path);
end

% 使用FieldTrip通道选择功能
cfg = [];
cfg.channel = {'all', '-M1', '-M2', '-VEO', '-HEO', '-CB1', '-CB2'};
selected_channels = ft_channelselection(cfg.channel, hdr_labels);

% 创建通道掩码（保持原始顺序）
keep_mask = ismember(hdr_labels, selected_channels);
removed_channels = hdr_labels(~keep_mask);
hdr_labels = hdr_labels(keep_mask);


bad_prop_1 = 0.4;
thresh_1 = 3;
bad_prop_2 = 0.01;
thresh_2 = 30;

for sub_id = 1:n_sub
    %% Load Data
    name = unique_names{sub_id};
    
    % 查找该被试的所有文件
    file_list = dir(fullfile(data_dir, [name '_*.mat']));
    
    % 验证文件数量
    if numel(file_list) ~= 3
        error('Subject %s has %d files (expected 3)', name, numel(file_list));
    end
    dates = cellfun(@(x) regexp(x.name, '\d{8}', 'match', 'once'), ...
                   num2cell(file_list), 'UniformOutput', false);
    [~, sort_idx] = sort(dates);
    file_list = file_list(sort_idx);
    % 初始化数据结构
    data = struct();
    data.trial = {};
    data.time = {};
    data.label = hdr_labels(:); % 确保列向量
    data.hdr = struct();
    data.hdr.label = hdr_labels(:); % 设置hdr.label
    data.fsample = Fs;
    
    % 处理每个文件
    for f_idx = 1:numel(file_list)
        file_path = fullfile(data_dir, file_list(f_idx).name);
        file_data = load(file_path);
        vars = fieldnames(file_data);
        eeg_vars = vars(~cellfun(@isempty, regexp(vars, '_eeg\d+$', 'once')));
        first_var = eeg_vars{1};
        prefix = regexp(first_var, '^(.*)_eeg\d+$', 'tokens', 'once');
        prefix = prefix{1};
        % 处理每个trial
        for n = 1:15
            var_name = sprintf('%s_eeg%d', prefix, n);
            trial_data = file_data.(var_name);
            
            % 转置数据（如果数据是时间×通道）
            % trial_data = trial_data'; % 根据实际数据维度决定是否需要转置
            trial_data = trial_data(keep_mask, :);
            
            
            % 添加到数据结构
            data.trial{end+1} = trial_data; % 假设数据是通道×时间
            n_samples = size(trial_data, 2);
            data.time{end+1} = (0:n_samples-1)/Fs;
        end
    end


%     file_name = fullfile(data_dir, files(sub_id).name);
%     file_name = char(file_name);
%     cfg = [];
%     cfg.dataset = file_name;
%     cfg.channel = {'all', '-M1', '-M2', '-VEO', '-HEO', '-CB1', '-CB2'};
%     data = ft_preprocessing(cfg);
    %% Choose EEG channel
%     cfg = [];
%     cfg.channel = {'all', '-M1', '-M2', '-VEO', '-HEO', '-CB1', '-CB2'};
%     data = ft_channelselection(cfg, data);

    %% Downsample
    resamplefs = 125;
    cfg = [];
    cfg.resamplefs = resamplefs;
    cfg.detrend = 'no';
    data_downsampled = ft_resampledata(cfg, data);
    pre_info.resamplefs = cfg.resamplefs;

    %% Filter
    cfg=[];
    cfg.bpfilter = 'yes';
    bpfreq = [0.5, 47];
    cfg.bpfreq = bpfreq;
%     cfg.continuous = 'yes';
    % The default bpfiltord is 4. Consider changing it to 3 if error occurs.
%     cfg.bpfiltord = 3; 
    pre_info.bpfreq = cfg.bpfreq;
    data_filted = ft_preprocessing(cfg, data_downsampled);

    %% Divide trial
%     cfg = [];
% %     beg_s1 = [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204];
% %     end_s1 = [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359];
% %     beg_s2 = [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741];
% %     end_s2 = [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817];
% %     beg_s3 = [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888];
% %     end_s3 = [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066];
%     %% SEED Start and end point at 1000 hz
%     beg_s = [13500,290000,551000,784000,1050000,1262000,1484000,1748000,1993000,2287000,2551000,2812000,3072000,3335000,3599000] / 1000;
%     end_s = [262000,523000,757000,1022000,1235000,1457000,1721000,1964000,2258000,2524000,2786000,3045000,3307000,3573000,3805000] / 1000;
%     
%     session = regexp(files(sub_id).name, '_(\d+)\.', 'tokens', 'once');
%     session = session{1};
%     subject_id = regexp(files(sub_id).name, '(\d+)_', 'tokens', 'once');
%     subject_id = subject_id{1};
%     trl = [beg_s;end_s] * resamplefs;
%     disp(trl)
%     trl = int32(trl);
%     disp(trl)
%     fprintf("Session %s\n", session)
%     trl = transpose(trl);
%     trl = padarray(trl, [0,1], "post");
%     cfg.trl = trl;
%     data_trialed = ft_redefinetrial(cfg, data_filted);
    data_trialed = data_filted;

    %% 1st Interpolate bad channels

    load chn_coords
    load('chn_coords')
    keep_channels = [1, 3, 6, 14];

    bad_channel_type_1 = cell(1, 45);
    bad_channel_type_2 = cell(1, 45);
    if debug
        visual_check_data(data_trialed.trial{5}, data_trialed.hdr.label, 50, 0, 'before process', resamplefs);
    end
    for i = 1: length(data_trialed.trial)
%         visual_check_data(data_trialed.trial{i}, data_trialed.hdr.label, 50, 0, 'before interpolation', resamplefs);
%         pause(40);
        x = transpose(data_trialed.trial{i}); % change to (n_points, n_channs)
        [iBad_1, toGood_1] = nt_find_bad_channels_custom(x, bad_prop_1, thresh_1);
        [iBad_2, toGood_2] = nt_find_bad_channels_custom(x, bad_prop_2, thresh_2);
        bad_channel_type_1{i} = union(bad_channel_type_1{i}, setdiff(iBad_1, keep_channels));
        bad_channel_type_2{i} = union(bad_channel_type_2{i}, setdiff(iBad_2, keep_channels));
        iBad = union(iBad_1, iBad_2);
        % keep [1, 3, 6, 14] (Fp1, Fp2, F7, F8) at first interpolation
        iBad = setdiff(iBad, keep_channels);
        if iBad
            pre_info.iBad{i} = iBad;
            fprintf('trial %d, iBad:', i)
            disp(iBad)
            disp(data_trialed.hdr.label(iBad))
%             visual_check_data(data.trial{i}, hdr.label, 50, 0, ['trial ', num2str(i)], resamplefs)
        else
            fprintf('1st bad_channel inter - Trial %d No bad channels identified\n\n', i)
        end
        SEED_coords_matrix = load('SEED_coords_matrix.mat').coords_matrix;
        if iBad
            [toGood,fromGood] = nt_interpolate_bad_channels_custom(x, iBad, SEED_coords_matrix);
            x = x * (toGood * fromGood);
        end
        data_trialed.trial{i} = transpose(x);
    end

    %% Auto ICA
    % IClabel: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other.
    params_ICLabel_def = [0 0;0.8 1; 0 0; 0 0; 0 0; 0 0; 0 0]; % 使用默认阈值
    chanlocs = readlocs('C:\Emotion_Classification\AutoICA\SEED_10_20_standard.ced');
    data_ica = data_trialed; % 初始化处理后的数据结构
    
    for trial_idx = 1:numel(data_trialed.trial)
        % ========== 当前trial数据准备 ==========
        current_trial = data_trialed.trial{trial_idx};
        
        % ========== 创建EEGLAB数据结构 ==========
        EEGtemp = eeg_emptyset();
        EEGtemp.setname = sprintf('Sub%d_Trial%d', sub_id, trial_idx);
        EEGtemp.filename = EEGtemp.setname;
        EEGtemp.filepath = fileparts(data_dir);
        EEGtemp.data = current_trial; % 直接使用当前trial数据
        EEGtemp.nbchan = size(EEGtemp.data, 1);
        EEGtemp.pnts = size(EEGtemp.data, 2);
        EEGtemp.trials = 1;
        EEGtemp.srate = resamplefs;
        EEGtemp.xmin = 0;
        EEGtemp.xmax = (EEGtemp.pnts - 1)/EEGtemp.srate;
        EEGtemp.chanlocs = chanlocs;
        
        % ========== 执行ICA ==========
        % 独立运行ICA
        EEGtemp = pop_runica(EEGtemp, 'icatype', 'runica', 'concatcond', 'off');
        
        % ========== 成分标记 ==========
        params_IClabel = params_ICLabel_def;
        EEGtemp = pop_iclabel(EEGtemp, 'default');
        EEGtemp = pop_icflag(EEGtemp, params_ICLabel_def);
        classifications = EEGtemp.etc.ic_classification.ICLabel.classifications; % Keep classifications before component substraction
%         for i = 1:size(EEGtemp.icaweights,1)
%           if(EEGtemp.etc.ic_classification.ICLabel.classifications(i, 3) > params_IClabel(3, 1))
%               % 如果被识别为眼电
%               if(EEGtemp.icaweights(i, 6) * EEGtemp.icaweights(i, 7) < 0)
%                   %如果fp1和fp2异号，即为水平眼动而非眨眼
%                   if(abs(EEGtemp.icaweights(i, 6) - EEGtemp.icaweights(i, 7)) > 0.25)
%                       % 如果左右差异大于阈值0.25（确定为水平眼动，而非噪声水平的左右不均衡）
%                       disp('tirggered!');
%                       % 保留该水平眼动成分
%                       EEGtemp.reject.gcompreject(i) = 1;
%                   end
%               end
%           end
%         end
        
        % ========== 可视化调试 ==========
        if debug
            figure('Name', EEGtemp.setname)
            pop_eegplot(EEGtemp, 1, 1, 1);
            pop_viewprops(EEGtemp, 0);
        end
        
        % ========== 去除噪声成分 ==========
        bad_components = find(EEGtemp.reject.gcompreject);
        disp('Classified as noise components:')
        disp(find(EEGtemp.reject.gcompreject == 1));
        disp(EEGtemp.etc.ic_classification.ICLabel.classifications)
        EEGtemp = pop_subcomp(EEGtemp,[], debug); % Subtract artifactual independent components
        EEGtemp.etc.ic_classification.ICLabel.orig_classifications = classifications;
        
        % ========== 存储处理后的数据 ==========
        data_ica.trial{trial_idx} = EEGtemp.data;
        
        % ========== 保存中间结果 ==========
        % 可选：保存每个trial的ICA结果
        % save(fullfile(save_dir, EEGtemp.setname), 'EEGtemp', '-v7.3');
    end
% 
%     EEGtemp = [];
%     EEGtemp.filepath = sprintf('C:\Emotion Classification\prep_code_clPaper\Processed\subject_%d\trial_%d', sub_id, i);
%     EEGtemp.filename = sprintf('subject_%d_trial_%d', sub_id, i);
%     EEGtemp.data = cat(2, data_trialed.trial{:});
%     %           visual_check_data(EEGtemp.data, hdr.label, 50, 0, 'Before-ICA', resamplefs);
%     EEGtemp.etc = [];
%     EEGtemp.setname = [];
%     EEGtemp.icawinv = [];
%     EEGtemp.icaweights = [];
%     EEGtemp.icasphere = [];
%     EEGtemp.nbchan = size(EEGtemp.data, 1);
%     EEGtemp.pnts = size(EEGtemp.data, 2);
%     EEGtemp.trials = 1;
%     EEGtemp.srate = resamplefs;
%     EEGtemp.xmin = 0;
%     EEGtemp.xmax = (EEGtemp.pnts - 1) / EEGtemp.srate;
%     EEGtemp.chanlocs = readlocs('C:\Emotion_Classification\AutoICA\SEED_10_20_standard.ced');
%     %               visual_check_data(EEGtemp.data, hdr.label, 50, 0, ['trial ', num2str(i)], resamplefs)
%     EEGtemp = pop_runica(EEGtemp,'icatype','runica','concatcond','off');
%     EEGtemp = pop_iclabel(EEGtemp,'default');
%     if(debug)
%         pop_eegplot(EEGtemp, 0);
%         pop_viewprops(EEGtemp, 0);
%     end
%     %             IClabel: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other.
%     params_ICLabel_cus = [0 0;0.5 1; 0.7 1; 0.7 1; 0.7 1; 0.7 1; 0 0];
%     params_ICLabel_def = [0 0;0.8 1; 0.8 1; 0 0; 0 0; 0 0; 0 0];
%     % which threshold to use
%     params_IClabel = params_ICLabel_def;
%     thresh = 'Def';
% %     params_IClabel = params_ICLabel_cus;
% %     thresh = 'Cus';
%     EEGtemp = pop_icflag(EEGtemp, params_IClabel);
%     
%     classifications = EEGtemp.etc.ic_classification.ICLabel.classifications; % Keep classifications before component substraction
%     
%     disp('Classified as noise components:')
%     disp(find(EEGtemp.reject.gcompreject == 1));
%     
%     disp(EEGtemp.etc.ic_classification.ICLabel.classifications)
%     EEGtemp = pop_subcomp(EEGtemp,[], debug); % Subtract artifactual independent components
%     EEGtemp.etc.ic_classification.ICLabel.orig_classifications = classifications;
%     
%     % binary classification
%     binary_matrix = zeros(size(classifications));
%     for i = 1:size(classifications, 1)
%         [max_values, max_indices] = max(classifications(i, :));
%         if(max_values > params_IClabel(max_indices))
%             binary_matrix(i, max_indices) = 1;
%         end
%     end
%     classification_count = sum(binary_matrix, 1);
% 
%     %% Divide trial and 2nd bad channel interpolate
% 
%     %  Divide
%     data_ica = data_trialed;
%     start_time = 1;
%     end_time = 0;
%     time_points = cellfun(@(x) size(x, 2), data.trial);
%     for vid_id = 1:45
%         end_time = start_time + time_points(vid_id);
%         disp(start_time);
%         disp(end_time);
%         data_ica.trial{vid_id} = EEGtemp.data(:, start_time:end_time);
%         start_time = end_time + 1;
%     end

    %  bad channel interpolate

    for i = 1: length(data_ica.trial)
%         visual_check_data(data_trialed.trial{i}, data_trialed.hdr.label, 50, 0, 'before interpolation', resamplefs);
%         pause(40);
        x = transpose(data_ica.trial{i}); % change to (n_points, n_channs)
        [iBad_1, toGood_1] = nt_find_bad_channels_custom(x, bad_prop_1, thresh_1);
        [iBad_2, toGood_2] = nt_find_bad_channels_custom(x, bad_prop_2, thresh_2);
        bad_channel_type_1{i} = union(bad_channel_type_1{i}, iBad_1);
        bad_channel_type_2{i} = union(bad_channel_type_2{i}, iBad_2);
        iBad = union(iBad_1, iBad_2);
        if iBad
            pre_info.iBad{i} = iBad;
            fprintf('trial %d, iBad:', i)
            disp(iBad)
            disp(data_ica.hdr.label(iBad))
%             visual_check_data(data.trial{i}, hdr.label, 50, 0, ['trial ', num2str(i)], resamplefs)
        else
            fprintf('2st bad_channel inter - Trial %d No bad channels identified\n\n', i)
        end
        SEED_coords_matrix = load('SEED_coords_matrix.mat').coords_matrix;
        if iBad
            [toGood,fromGood] = nt_interpolate_bad_channels_custom(x, iBad, SEED_coords_matrix);
            x = x * (toGood * fromGood);
        end
        data_ica.trial{i} = transpose(x);
    end
    if(0)
        visual_check_data(data_ica.trial{i}, data_ica.hdr.label, 50, 0, 'After 2nd interpolation', resamplefs);
%         pause(40);
    end

    %% Rereference
    nomas_ind = 1:58;
    for i = 1: length(data_ica.trial)
        data_ica.trial{i} = data_ica.trial{i}(nomas_ind, :) - repmat(mean(data_ica.trial{i}(nomas_ind, :)), 58, 1);
    end

    %% Reorder trials
%     trial_reorder = data.trial;
%     trial_reorder(num_reorder) = data.trial;
%     for i = 1:28
%         n_samples(sub,i) = size(trial_reorder{1,i}, 2) / resamplefs;
%     end
%     disp(n_samples(sub,:))
%     pre_info.num_reorder = num_reorder;
%     pre_info.classification = classification_count;
%     pre_info.noise_sum = sum(pre_info.classification(2:6));
%     pre_info.bad_channel_1 = bad_channel_type_1;
%     pre_info.bad_channel_2 = bad_channel_type_2;
    if(debug)
        visual_check_data(data_ica.trial{5}, data_ica.hdr.label(nomas_ind), 50, 0, 'after process', resamplefs);
    end

    %% Save data
    
    n_samples_one = zeros(1, 45);
    data_all_cleaned = zeros(58, 0);
    for i = 1: length(data_ica.trial)
        data_all_cleaned = cat(2, data_all_cleaned, data_ica.trial{i});
        n_samples_one(1, i) = size(data_ica.trial{i}, 2) / resamplefs;
    end
    thresh = 'def';
    save_folder_name = sprintf('Processed_data_filter_%.2f_%.2f_AutoICA_%s_Threshold', bpfreq(1), bpfreq(2), thresh);
    save_dir = fullfile('C:\Emotion Classification\Processed\SEED', save_folder_name, 'data');
    if ~exist(save_dir)
        mkdir(save_dir);
    end
    save(fullfile(save_dir, sprintf('Sub_%d_all', sub_id)), 'data_all_cleaned', 'n_samples_one');
end

