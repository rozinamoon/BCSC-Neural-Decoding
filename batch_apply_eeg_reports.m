%% BATCH APPLY MANUAL EEG REJECTION REPORTS
% Starting point:
%   S##_4_set_trials[_c|_tl|_tt].set
%
% For each subject/category, this script:
%   1) removes first-pass bad trials from the report
%   2) runs ICA #1 and removes reported components
%   3) removes second-pass bad trials
%   4) runs ICA #2 and removes reported components
%   5) saves S##_10_remove_bad_components_2_<suffix>.set/.fdt
%
% Missing report/input files are skipped. Incomplete reports are skipped by
% default so that an apparently successful but incorrect final dataset is
% not created.

clear;
clc;

%% USER SETTINGS

DATA_ROOT = '/Users/moonrozina/BCSC-Neural-Decoding/data/EEG_Data_Processing_ses02';
SUBJECT_NUMBERS = 1:16;

% false: keep an existing final dataset and skip it
% true : delete and rebuild the existing final .set/.fdt
OVERWRITE_EXISTING_FINAL = false;

% The original completed report should contain all four labeled lines.
% Keep false for safety. Set true only when you intentionally want missing
% report fields to be interpreted as empty rejection lists.
ALLOW_INCOMPLETE_REPORT = false;

% Prefer the exact saved ICA decomposition datasets when they exist.
% This is scientifically safer because report component numbers refer to the
% ICA decomposition used when the report was written.
PREFER_SAVED_ICA_STAGES = true;

% When an exact saved ICA stage is unavailable, the script can recompute ICA.
% Recomputed ICA can change component order, so the report component numbers
% may not identify exactly the same sources as in the original analysis.
ALLOW_ICA_RECOMPUTATION = true;
RNG_SEED_BASE = 20220829;

% Intermediate stage files are not required for the final result.
SAVE_INTERMEDIATE_STAGES = false;

% Requires the original plot_results.m on the MATLAB path.
MAKE_FINAL_FIGURE = false;

% ICA settings copied from the original script.
ICA_TYPE = 'runica';
ICA_EXTENDED = 1;

%% CHECK EEGLAB

if exist('pop_loadset', 'file') ~= 2 || exist('pop_runica', 'file') ~= 2
    error(['EEGLAB is not on the MATLAB path. Open EEGLAB first or add ', ...
           'the EEGLAB folder to the MATLAB path, then run this script again.']);
end

%% CATEGORY DEFINITIONS

categories = struct( ...
    'name', { ...
        'Control', ...
        'TMS on lip target', ...
        'TMS on tongue target'}, ...
    'folder_candidates', { ...
        {'Control', 'control'}, ...
        {'TMS_on_Lip_Targuet', 'TMS_on_Lip_Target', 'tms_true_lip'}, ...
        {'TMS_on_Tongue_Target', 'tms_true_tongue'}}, ...
    'suffix', { ...
        '_c', ...
        '_tl', ...
        '_tt'} ...
);

summary_rows = {};
processed_count = 0;
skipped_count = 0;
failed_count = 0;

fprintf('\nData root: %s\n', DATA_ROOT);
fprintf('Subjects: S%02d-S%02d\n\n', SUBJECT_NUMBERS(1), SUBJECT_NUMBERS(end));

%% MAIN LOOP

for category_index = 1:numel(categories)
    spec = categories(category_index);
    category_folder = find_existing_folder(DATA_ROOT, spec.folder_candidates);

    if isempty(category_folder)
        warning('Category folder not found for "%s". Skipping category.', spec.name);
        for subject_number = SUBJECT_NUMBERS
            subject = sprintf('S%02d', subject_number);
            summary_rows(end+1, :) = {subject, spec.name, 'SKIPPED', ...
                'Category folder missing', '', '', ''}; %#ok<SAGROW>
            skipped_count = skipped_count + 1;
        end
        continue;
    end

    fprintf('=== %s ===\n%s\n', spec.name, category_folder);

    for subject_number = SUBJECT_NUMBERS
        subject = sprintf('S%02d', subject_number);

        report_path = first_existing_file({ ...
            fullfile(category_folder, [subject '_report' spec.suffix '.txt']), ...
            fullfile(category_folder, [subject '_report.txt']) ...
        });

        input_set_path = first_existing_file({ ...
            fullfile(category_folder, [subject '_4_set_trials' spec.suffix '.set']), ...
            fullfile(category_folder, [subject '_4_set_trials.set']) ...
        });

        output_base = [subject '_10_remove_bad_components_2' spec.suffix];
        output_set_path = fullfile(category_folder, [output_base '.set']);
        existing_final_path = first_existing_file({ ...
            output_set_path, ...
            fullfile(category_folder, [subject '_10_remove_bad_components_2.set']) ...
        });

        if isempty(report_path)
            fprintf('[SKIP] %s: report not found.\n', subject);
            summary_rows(end+1, :) = {subject, spec.name, 'SKIPPED', ...
                'Report missing', '', input_set_path, output_set_path}; %#ok<SAGROW>
            skipped_count = skipped_count + 1;
            continue;
        end

        if isempty(input_set_path)
            fprintf('[SKIP] %s: stage-4 .set file not found.\n', subject);
            summary_rows(end+1, :) = {subject, spec.name, 'SKIPPED', ...
                'Stage-4 input missing', report_path, '', output_set_path}; %#ok<SAGROW>
            skipped_count = skipped_count + 1;
            continue;
        end

        if ~isempty(existing_final_path) && ~OVERWRITE_EXISTING_FINAL
            fprintf('[SKIP] %s: final dataset already exists (%s).\n', ...
                subject, existing_final_path);
            summary_rows(end+1, :) = {subject, spec.name, 'SKIPPED', ...
                'Final output already exists', report_path, input_set_path, existing_final_path}; %#ok<SAGROW>
            skipped_count = skipped_count + 1;
            continue;
        end

        try
            report = parse_rejection_report(report_path);

            if ~report.is_complete && ~ALLOW_INCOMPLETE_REPORT
                missing_text = strjoin(report.missing_fields, ', ');
                fprintf('[SKIP] %s: incomplete report (%s).\n', subject, missing_text);
                summary_rows(end+1, :) = {subject, spec.name, 'SKIPPED', ...
                    ['Incomplete report: ' missing_text], report_path, input_set_path, output_set_path}; %#ok<SAGROW>
                skipped_count = skipped_count + 1;
                continue;
            end

            fprintf('\n[RUN ] %s\n', subject);
            print_report_lists(report);

            [input_folder, input_name, input_ext] = fileparts(input_set_path);
            EEG = pop_loadset('filename', [input_name input_ext], 'filepath', input_folder);
            EEG = eeg_checkset(EEG);

            %% FIRST BAD-TRIAL REMOVAL
            EEG = reject_epochs_checked(EEG, report.bad_trials_1, ...
                sprintf('%s %s first bad-trial removal', subject, spec.name));

            if SAVE_INTERMEDIATE_STAGES
                EEG = save_dataset(EEG, category_folder, ...
                    [subject '_5_remove_bad_trials' spec.suffix], true);
            end

            %% ICA #1 AND COMPONENT REMOVAL
            saved_ica_1_path = first_existing_file({ ...
                fullfile(category_folder, [subject '_6_ica_decomposition' spec.suffix '.set']), ...
                fullfile(category_folder, [subject '_6_ica_decomposition.set']) ...
            });

            [EEG, used_saved_ica_1] = obtain_ica_dataset( ...
                EEG, saved_ica_1_path, PREFER_SAVED_ICA_STAGES, ...
                ALLOW_ICA_RECOMPUTATION, ICA_TYPE, ICA_EXTENDED, ...
                RNG_SEED_BASE + subject_number * 10 + 1, ...
                sprintf('%s %s ICA #1', subject, spec.name));

            if SAVE_INTERMEDIATE_STAGES && ~used_saved_ica_1
                EEG = save_dataset(EEG, category_folder, ...
                    [subject '_6_ica_decomposition' spec.suffix], true);
            end

            EEG = remove_components_checked(EEG, report.bad_components_1, ...
                sprintf('%s %s first ICA removal', subject, spec.name));

            if SAVE_INTERMEDIATE_STAGES
                EEG = save_dataset(EEG, category_folder, ...
                    [subject '_7_remove_bad_components' spec.suffix], true);
            end

            %% SECOND BAD-TRIAL REMOVAL
            % These indices are applied AFTER the first trial removal and
            % first ICA component removal, matching the original workflow.
            EEG = reject_epochs_checked(EEG, report.bad_trials_2, ...
                sprintf('%s %s second bad-trial removal', subject, spec.name));

            if SAVE_INTERMEDIATE_STAGES
                EEG = save_dataset(EEG, category_folder, ...
                    [subject '_8_remove_bad_trials_2' spec.suffix], true);
            end

            %% ICA #2 AND COMPONENT REMOVAL
            saved_ica_2_path = first_existing_file({ ...
                fullfile(category_folder, [subject '_9_ica_decomposition_2' spec.suffix '.set']), ...
                fullfile(category_folder, [subject '_9_ica_decomposition_2.set']) ...
            });

            [EEG, used_saved_ica_2] = obtain_ica_dataset( ...
                EEG, saved_ica_2_path, PREFER_SAVED_ICA_STAGES, ...
                ALLOW_ICA_RECOMPUTATION, ICA_TYPE, ICA_EXTENDED, ...
                RNG_SEED_BASE + subject_number * 10 + 2, ...
                sprintf('%s %s ICA #2', subject, spec.name));

            if SAVE_INTERMEDIATE_STAGES && ~used_saved_ica_2
                EEG = save_dataset(EEG, category_folder, ...
                    [subject '_9_ica_decomposition_2' spec.suffix], true);
            end

            EEG = remove_components_checked(EEG, report.bad_components_2, ...
                sprintf('%s %s second ICA removal', subject, spec.name));

            %% FINAL SAVE
            if OVERWRITE_EXISTING_FINAL
                if ~isempty(existing_final_path)
                    delete_existing_dataset(existing_final_path);
                end
                if ~strcmp(existing_final_path, output_set_path)
                    delete_existing_dataset(output_set_path);
                end
            end

            EEG = save_dataset(EEG, category_folder, output_base, false);

            if MAKE_FINAL_FIGURE
                if exist('plot_results', 'file') == 2
                    stage = "Final Result";
                    save_file = fullfile(category_folder, [subject '_final' spec.suffix]);
                    plot_results(EEG, 1, 1, 1, 1, save_file, stage, "True");
                    close all;
                else
                    warning('plot_results.m was not found. Final figure was not created.');
                end
            end

            fprintf('[DONE] %s -> %s\n\n', subject, output_set_path);
            summary_rows(end+1, :) = {subject, spec.name, 'PROCESSED', ...
                'OK', report_path, input_set_path, output_set_path}; %#ok<SAGROW>
            processed_count = processed_count + 1;

            clear EEG;

        catch ME
            fprintf(2, '[FAIL] %s: %s\n\n', subject, ME.message);
            summary_rows(end+1, :) = {subject, spec.name, 'FAILED', ...
                ME.message, report_path, input_set_path, output_set_path}; %#ok<SAGROW>
            failed_count = failed_count + 1;
            clear EEG;
        end
    end

    fprintf('\n');
end

%% WRITE SUMMARY

summary_table = cell2table(summary_rows, 'VariableNames', { ...
    'Subject', 'Category', 'Status', 'Message', ...
    'ReportFile', 'InputSetFile', 'OutputSetFile'});

summary_file = fullfile(DATA_ROOT, 'batch_apply_reports_summary.csv');
writetable(summary_table, summary_file);

fprintf('========================================\n');
fprintf('Processed: %d\n', processed_count);
fprintf('Skipped:   %d\n', skipped_count);
fprintf('Failed:    %d\n', failed_count);
fprintf('Summary:   %s\n', summary_file);
fprintf('========================================\n');

%% LOCAL FUNCTIONS

function folder_path = find_existing_folder(root_folder, folder_candidates)
    folder_path = '';
    for index = 1:numel(folder_candidates)
        candidate = fullfile(root_folder, folder_candidates{index});
        if isfolder(candidate)
            folder_path = candidate;
            return;
        end
    end
end

function file_path = first_existing_file(candidates)
    file_path = '';
    for index = 1:numel(candidates)
        if isfile(candidates{index})
            file_path = candidates{index};
            return;
        end
    end
end

function report = parse_rejection_report(report_path)
    report_text = fileread(report_path);

    [line_1, found_1] = extract_labeled_line(report_text, { ...
        'Removed trials in the first inspection'});

    [line_2, found_2] = extract_labeled_line(report_text, { ...
        'Removed ICA components in the first inspection'});

    [line_3, found_3] = extract_labeled_line(report_text, { ...
        'Removed trials in the second inspection'});

    % The original code writes "Removed ICA trials..." here, even though
    % these values are ICA component indices. Accept both spellings.
    [line_4, found_4] = extract_labeled_line(report_text, { ...
        'Removed ICA components in the second inspection', ...
        'Removed ICA trials in the second inspection'});

    report.bad_trials_1 = parse_index_list(line_1);
    report.bad_components_1 = parse_index_list(line_2);
    report.bad_trials_2 = parse_index_list(line_3);
    report.bad_components_2 = parse_index_list(line_4);

    found_flags = [found_1, found_2, found_3, found_4];
    field_names = { ...
        'first bad-trial line', ...
        'first ICA-component line', ...
        'second bad-trial line', ...
        'second ICA-component line'};

    report.is_complete = all(found_flags);
    report.missing_fields = field_names(~found_flags);
end

function [line_value, found] = extract_labeled_line(report_text, labels)
    line_value = '';
    found = false;

    for label_index = 1:numel(labels)
        escaped_label = regexptranslate('escape', labels{label_index});
        pattern = ['(?mi)^\s*' escaped_label '\s*:\s*(.*?)\s*$'];
        token = regexp(report_text, pattern, 'tokens', 'once');

        if ~isempty(token)
            line_value = strtrim(token{1});
            found = true;
            return;
        end
    end
end

function indices = parse_index_list(raw_text)
    raw_text = strtrim(raw_text);

    if isempty(raw_text) || any(strcmpi(raw_text, {'none', 'nan', '[]', '-', 'n/a'}))
        indices = [];
        return;
    end

    % Supports formats such as:
    %   [1, 3, 7]
    %   1 3 7
    %   1:4
    %   1:2:9
    tokens = regexp(raw_text, '\d+(?::\d+(?::\d+)?)?', 'match');
    indices = [];

    for token_index = 1:numel(tokens)
        token = tokens{token_index};
        colon_parts = strsplit(token, ':');
        numbers = str2double(colon_parts);

        if numel(numbers) == 1
            expanded = numbers;
        elseif numel(numbers) == 2
            expanded = numbers(1):numbers(2);
        elseif numel(numbers) == 3
            expanded = numbers(1):numbers(2):numbers(3);
        else
            error('Could not parse index token: %s', token);
        end

        indices = [indices, expanded]; %#ok<AGROW>
    end

    if isempty(indices)
        error('The report line is not empty, but no numeric indices could be parsed: "%s"', raw_text);
    end

    if any(~isfinite(indices)) || any(indices < 1) || any(mod(indices, 1) ~= 0)
        error('Report indices must be positive integers: "%s"', raw_text);
    end

    indices = unique(indices, 'stable');
end

function [EEG, used_saved_stage] = obtain_ica_dataset( ...
        current_EEG, saved_stage_path, prefer_saved, allow_recompute, ...
        ica_type, ica_extended, rng_seed, stage_name)

    used_saved_stage = false;

    if prefer_saved && ~isempty(saved_stage_path)
        [saved_folder, saved_name, saved_ext] = fileparts(saved_stage_path);
        candidate_EEG = pop_loadset( ...
            'filename', [saved_name saved_ext], ...
            'filepath', saved_folder);
        candidate_EEG = eeg_checkset(candidate_EEG);

        if datasets_dimensionally_compatible(current_EEG, candidate_EEG) && ...
                ~isempty(candidate_EEG.icaweights)
            fprintf('  %s: using saved ICA decomposition %s\n', ...
                stage_name, saved_stage_path);
            EEG = candidate_EEG;
            used_saved_stage = true;
            return;
        else
            warning(['%s: saved ICA stage exists but its dimensions/ICA weights ', ...
                     'do not match the current dataset. ICA will be recomputed.'], ...
                    stage_name);
        end
    end

    if ~allow_recompute
        error(['%s: no compatible saved ICA stage was found, and ', ...
               'ALLOW_ICA_RECOMPUTATION is false.'], stage_name);
    end

    warning(['%s: recomputing ICA. The report component numbers were selected ', ...
             'from the original ICA decomposition; a new ICA run may reorder ', ...
             'components and therefore may not reproduce the exact original removal.'], ...
            stage_name);

    rng(rng_seed, 'twister');
    EEG = pop_runica(current_EEG, ...
        'icatype', ica_type, ...
        'extended', ica_extended, ...
        'interrupt', 'off');
    EEG = eeg_checkset(EEG);
end

function compatible = datasets_dimensionally_compatible(EEG_a, EEG_b)
    compatible = ...
        EEG_a.nbchan == EEG_b.nbchan && ...
        EEG_a.pnts == EEG_b.pnts && ...
        EEG_a.trials == EEG_b.trials && ...
        abs(double(EEG_a.srate) - double(EEG_b.srate)) < eps(max(double(EEG_a.srate), 1));
end

function EEG = reject_epochs_checked(EEG, indices, stage_name)
    if isempty(indices)
        fprintf('  %s: no trials to remove.\n', stage_name);
        return;
    end

    trial_count = EEG.trials;
    invalid = indices(indices < 1 | indices > trial_count);

    if ~isempty(invalid)
        error(['%s contains trial indices outside 1-%d: %s. ', ...
               'The report must be applied to the matching stage-4 dataset.'], ...
              stage_name, trial_count, mat2str(invalid));
    end

    fprintf('  %s: removing trials %s\n', stage_name, mat2str(indices));
    EEG = pop_rejepoch(EEG, indices, 0);
    EEG = eeg_checkset(EEG);
end

function EEG = remove_components_checked(EEG, indices, stage_name)
    if isempty(EEG.icaweights)
        error('%s: ICA weights are missing.', stage_name);
    end

    component_count = size(EEG.icaweights, 1);

    if isempty(indices)
        fprintf('  %s: no ICA components to remove.\n', stage_name);
        return;
    end

    invalid = indices(indices < 1 | indices > component_count);

    if ~isempty(invalid)
        error('%s contains component indices outside 1-%d: %s.', ...
              stage_name, component_count, mat2str(invalid));
    end

    fprintf('  %s: removing ICA components %s\n', stage_name, mat2str(indices));
    EEG = pop_subcomp(EEG, indices, 0);
    EEG = eeg_checkset(EEG);
end

function EEG = save_dataset(EEG, folder_path, output_base, overwrite_existing)
    output_set_path = fullfile(folder_path, [output_base '.set']);

    if overwrite_existing
        delete_existing_dataset(output_set_path);
    elseif isfile(output_set_path)
        error('Output already exists: %s', output_set_path);
    end

    EEG.setname = output_base;
    EEG = pop_saveset(EEG, ...
        'filename', [output_base '.set'], ...
        'filepath', folder_path, ...
        'savemode', 'twofiles');
    EEG = eeg_checkset(EEG);
end

function delete_existing_dataset(set_path)
    if isfile(set_path)
        delete(set_path);
    end

    [folder_path, base_name, ~] = fileparts(set_path);
    fdt_path = fullfile(folder_path, [base_name '.fdt']);

    if isfile(fdt_path)
        delete(fdt_path);
    end
end

function print_report_lists(report)
    fprintf('  bad trials #1:     %s\n', display_indices(report.bad_trials_1));
    fprintf('  bad components #1: %s\n', display_indices(report.bad_components_1));
    fprintf('  bad trials #2:     %s\n', display_indices(report.bad_trials_2));
    fprintf('  bad components #2: %s\n', display_indices(report.bad_components_2));
end

function text_value = display_indices(indices)
    if isempty(indices)
        text_value = 'none';
    else
        text_value = mat2str(indices);
    end
end
