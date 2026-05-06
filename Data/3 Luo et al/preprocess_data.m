% preprocess_mat_files.m
%
% For each .mat file in the current folder:
%   1) Load the file (expects exactly one main variable, which is an N x 1 cell array)
%   2) Find a = min_i a_i, where size(cell_i) = [a_i, b, c_i]
%   3) Crop each cell to [a, b, c_i] by keeping the beginning
%   4) Concatenate all cropped cells along 3rd dimension -> X of size [a, b, sum_i c_i]
%   5) Save X into subfolder "preprocessed" as the same filename, with -v5
%   6) Build G of size [N, totalC, totalC], where each slice G(i,:,:) has ones on the
%      diagonal entries corresponding to the block of cell i
%   7) Save G into "preprocessed" as G_<filename>.mat, with -v5

clear;
clc;

inputFolder = pwd;  % change if needed
outputFolder = fullfile(inputFolder, 'preprocessed');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

files = dir(fullfile(inputFolder, '*.mat'));
nFiles = numel(files);

fprintf('Found %d .mat file(s).\n', nFiles);

for f = 1:nFiles
    fileName = files(f).name;
    filePath = fullfile(inputFolder, fileName);

    fprintf('\n[%d/%d] Processing %s ...\n', f, nFiles, fileName);

    S = load(filePath);
    varNames = fieldnames(S);

    if isempty(varNames)
        warning('File %s is empty. Skipping.', fileName);
        continue;
    end

    % If there are multiple variables, take the first one.
    % You can make this stricter if needed.
    data = S.(varNames{1});

    if ~iscell(data)
        warning('File %s: first variable is not a cell array. Skipping.', fileName);
        continue;
    end

    if isempty(data)
        warning('File %s: cell array is empty. Skipping.', fileName);
        continue;
    end

    nCells = numel(data);

    % Collect sizes
    aSizes = zeros(nCells, 1);
    bSizes = zeros(nCells, 1);
    cSizes = zeros(nCells, 1);

    valid = true;

    for i = 1:nCells
        Xi = data{i};

        if ~isnumeric(Xi) || ndims(Xi) ~= 3
            warning('File %s: cell %d is not a numeric 3D array. Skipping file.', fileName, i);
            valid = false;
            break;
        end

        sz = size(Xi);
        aSizes(i) = sz(1);
        bSizes(i) = sz(2);
        cSizes(i) = sz(3);
    end

    if ~valid
        continue;
    end

    % Check that b is the same for all cells
    if any(bSizes ~= bSizes(1))
        warning('File %s: second dimension b is not constant across cells. Skipping.', fileName);
        continue;
    end

    a = min(aSizes);
    b = bSizes(1);
    totalC = sum(cSizes);

    fprintf('  nCells = %d, min a = %d, b = %d, totalC = %d\n', nCells, a, b, totalC);

    % Crop and concatenate
    croppedCells = cell(nCells, 1);
    for i = 1:nCells
        Xi = data{i};
        croppedCells{i} = Xi(1:a, :, :);  % keep the beginning, crop from the end
    end

    X = cat(3, croppedCells{:});   % size [a, b, totalC]

    % Save concatenated tensor using same filename in preprocessed/
    outPathX = fullfile(outputFolder, fileName);
    save(outPathX, 'X', '-v6');
    fprintf('  Saved concatenated tensor to %s\n', outPathX);

    % Build G of size [nCells, totalC, totalC]
    G = zeros(nCells, totalC, totalC, 'uint8');

    startIdx = 1;
    for i = 1:nCells
        ci = cSizes(i);
        idx = startIdx:(startIdx + ci - 1);

        % Put ones on the diagonal of the block for slice i
        for k = idx
            G(i, k, k) = 1;
        end

        startIdx = startIdx + ci;
    end

    outPathG = fullfile(outputFolder, ['G_' fileName]);
    save(outPathG, 'G', '-v6');
    fprintf('  Saved block tensor G to %s\n', outPathG);
end

fprintf('\nDone.\n');f