classdef ECGFeatureExtractor
    properties
        sampling_rate
        prefix
        average
    end
    
    methods
        function obj = ECGFeatureExtractor(sampling_rate, prefix, average)
            if nargin < 2
                prefix = 'ecg';
            end
            if nargin < 3
                average = false;
            end
            obj.sampling_rate = sampling_rate;
            obj.prefix = prefix;
            obj.average = average;
        end
        
        function rr_int = get_RR_interval(~, peaks_locs, beatno, interval)
            rr_int = (peaks_locs(beatno + interval + 1) - peaks_locs(beatno + interval)) / obj.sampling_rate;
        end
        
        function rr_m = get_mean_RR(obj, peaks_locs, beatno)
            rr_intervals = [
                obj.get_RR_interval(peaks_locs, beatno, -1), ...
                obj.get_RR_interval(peaks_locs, beatno, 0), ...
                obj.get_RR_interval(peaks_locs, beatno, 1)
            ];
            rr_m = mean(rr_intervals);
        end
        
        function feature = get_diff(~, sig, loc_array1, loc_array2, beatno, amplitude)
            if amplitude
                feature = sig(loc_array2(beatno)) - sig(loc_array1(beatno));
            else
                feature = abs((loc_array2(beatno) - loc_array1(beatno))) / obj.sampling_rate;
            end
        end
        
        function features_rpeaks = from_Rpeaks(obj, sig, peaks_locs)
            if obj.sampling_rate <= 0
                error('Sampling rate must be greater than 0.');
            end

            FEATURES_RPEAKS = struct( ...
                'a_R', @(sig, ~, peaks_locs, beatno) sig(peaks_locs(beatno)), ...
                'RR0', @(~, sampling_rate, peaks_locs, beatno) obj.get_RR_interval(peaks_locs, beatno, -1), ...
                'RR1', @(~, sampling_rate, peaks_locs, beatno) obj.get_RR_interval(peaks_locs, beatno, 0), ...
                'RR2', @(~, sampling_rate, peaks_locs, beatno) obj.get_RR_interval(peaks_locs, beatno, 1), ...
                'RRm', @(~, sampling_rate, peaks_locs, beatno) obj.get_mean_RR(peaks_locs, beatno), ...
                'RR_0_1', @(~, sampling_rate, peaks_locs, beatno) obj.get_RR_interval(peaks_locs, beatno, -1) / ...
                    obj.get_RR_interval(peaks_locs, beatno, 0), ...
                'RR_2_1', @(~, sampling_rate, peaks_locs, beatno) obj.get_RR_interval(peaks_locs, beatno, 1) / ...
                    obj.get_RR_interval(peaks_locs, beatno, 0), ...
                'RR_m_1', @(~, sampling_rate, peaks_locs, beatno) obj.get_mean_RR(peaks_locs, beatno) / ...
                    obj.get_RR_interval(peaks_locs, beatno, 0) ...
            );

            features_rpeaks = struct();
            for m = 2:length(peaks_locs) - 2
                features = struct();
                keys = fieldnames(FEATURES_RPEAKS);
                for i = 1:length(keys)
                    key = keys{i};
                    func = FEATURES_RPEAKS.(key);
                    try
                        features.([obj.prefix, '_', key]) = func(sig, obj.sampling_rate, peaks_locs, m);
                    catch
                        features.([obj.prefix, '_', key]) = NaN;
                    end
                end
                features_rpeaks(m) = features;
            end

            if obj.average
                features_avr = struct();
                keys = fieldnames(features_rpeaks(2));
                for i = 1:length(keys)
                    key = keys{i};
                    values = arrayfun(@(x) x.(key), features_rpeaks, 'UniformOutput', false);
                    features_avr.(key) = mean(cell2mat(values));
                end
                features_rpeaks = features_avr;
            end
        end
        
        function features_waves = from_waves(obj, sig, R_peaks, fiducials)
            if obj.sampling_rate <= 0
                error('Sampling rate must be greater than 0.');
            end

            FEATURES_WAVES = struct( ...
                't_PR', @(sig, sampling_rate, locs_P, ~, locs_R, ~, ~, beatno) obj.get_diff(sig, locs_P, locs_R, beatno, false), ...
                't_QR', @(sig, sampling_rate, ~, locs_Q, locs_R, ~, ~, beatno) obj.get_diff(sig, locs_Q, locs_R, beatno, false), ...
                't_RS', @(sig, sampling_rate, ~, ~, locs_R, locs_S, ~, beatno) obj.get_diff(sig, locs_S, locs_R, beatno, false), ...
                't_RT', @(sig, sampling_rate, ~, ~, locs_R, ~, locs_T, beatno) obj.get_diff(sig, locs_T, locs_R, beatno, false), ...
                't_PQ', @(sig, sampling_rate, locs_P, locs_Q, ~, ~, ~, beatno) obj.get_diff(sig, locs_P, locs_Q, beatno, false), ...
                't_PS', @(sig, sampling_rate, locs_P, ~, ~, locs_S, ~, beatno) obj.get_diff(sig, locs_P, locs_S, beatno, false), ...
                't_PT', @(sig, sampling_rate, locs_P, ~, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_P, locs_T, beatno, false), ...
                't_QS', @(sig, sampling_rate, ~, locs_Q, ~, locs_S, ~, beatno) obj.get_diff(sig, locs_Q, locs_S, beatno, false), ...
                't_QT', @(sig, sampling_rate, ~, locs_Q, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_Q, locs_T, beatno, false), ...
                't_ST', @(sig, sampling_rate, ~, ~, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_S, locs_T, beatno, false), ...
                't_PT_QS', @(sig, sampling_rate, locs_P, locs_Q, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_P, locs_T, beatno, false) / ...
                    obj.get_diff(sig, locs_Q, locs_S, beatno, false), ...
                't_QT_QS', @(sig, sampling_rate, ~, locs_Q, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_Q, locs_T, beatno, false) / ...
                    obj.get_diff(sig, locs_Q, locs_S, beatno, false), ...
                'a_PQ', @(sig, sampling_rate, locs_P, locs_Q, ~, ~, ~, beatno) obj.get_diff(sig, locs_P, locs_Q, beatno, true), ...
                'a_QR', @(sig, sampling_rate, ~, locs_Q, locs_R, ~, ~, beatno) obj.get_diff(sig, locs_Q, locs_R, beatno, true), ...
                'a_RS', @(sig, sampling_rate, ~, ~, locs_R, locs_S, ~, beatno) obj.get_diff(sig, locs_R, locs_S, beatno, true), ...
                'a_ST', @(sig, sampling_rate, ~, ~, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_S, locs_T, beatno, true), ...
                'a_PS', @(sig, sampling_rate, locs_P, ~, ~, locs_S, ~, beatno) obj.get_diff(sig, locs_P, locs_S, beatno, true), ...
                'a_PT', @(sig, sampling_rate, locs_P, ~, ~, locs_T, beatno) obj.get_diff(sig, locs_P, locs_T, beatno, true), ...
                'a_QS', @(sig, sampling_rate, ~, locs_Q, ~, locs_S, ~, beatno) obj.get_diff(sig, locs_Q, locs_S, beatno, true), ...
                'a_QT', @(sig, sampling_rate, ~, locs_Q, ~, ~, locs_T, beatno) obj.get_diff(sig, locs_Q, locs_T, beatno, true), ...
                'a_ST_QS', @(sig, sampling_rate, ~, locs_Q, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_S, locs_T, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_S, beatno, true), ...
                'a_RS_QR', @(sig, sampling_rate, ~, locs_Q, locs_R, locs_S, ~, beatno) obj.get_diff(sig, locs_R, locs_S, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_R, beatno, true), ...
                'a_PQ_QS', @(sig, sampling_rate, locs_P, locs_Q, ~, locs_S, ~, beatno) obj.get_diff(sig, locs_P, locs_Q, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_S, beatno, true), ...
                'a_PQ_QT', @(sig, sampling_rate, locs_P, locs_Q, ~, ~, locs_T, beatno) obj.get_diff(sig, locs_P, locs_Q, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_T, beatno, true), ...
                'a_PQ_PS', @(sig, sampling_rate, locs_P, locs_Q, ~, locs_S, ~, beatno) obj.get_diff(sig, locs_P, locs_Q, beatno, true) / ...
                    obj.get_diff(sig, locs_P, locs_S, beatno, true), ...
                'a_PQ_QR', @(sig, sampling_rate, locs_P, locs_Q, locs_R, ~, ~, beatno) obj.get_diff(sig, locs_P, locs_Q, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_R, beatno, true), ...
                'a_PQ_RS', @(sig, sampling_rate, locs_P, locs_Q, locs_R, locs_S, ~, beatno) obj.get_diff(sig, locs_P, locs_Q, beatno, true) / ...
                    obj.get_diff(sig, locs_R, locs_S, beatno, true), ...
                'a_RS_QS', @(sig, sampling_rate, ~, locs_Q, locs_R, locs_S, ~, beatno) obj.get_diff(sig, locs_R, locs_S, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_S, beatno, true), ...
                'a_RS_QT', @(sig, sampling_rate, ~, locs_Q, locs_R, locs_S, locs_T, beatno) obj.get_diff(sig, locs_R, locs_S, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_T, beatno, true), ...
                'a_ST_PQ', @(sig, sampling_rate, locs_P, locs_Q, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_S, locs_T, beatno, true) / ...
                    obj.get_diff(sig, locs_P, locs_Q, beatno, true), ...
                'a_ST_QT', @(sig, sampling_rate, ~, locs_Q, ~, locs_S, locs_T, beatno) obj.get_diff(sig, locs_S, locs_T, beatno, true) / ...
                    obj.get_diff(sig, locs_Q, locs_T, beatno, true) ...
            );

            fiducial_names = ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"];
            fiducials = rmfield(fiducials, setdiff(fieldnames(fiducials), fiducial_names));

            P_peaks = fiducials.ECG_P_Peaks;
            Q_peaks = fiducials.ECG_Q_Peaks;
            S_peaks = fiducials.ECG_S_Peaks;
            T_peaks = fiducials.ECG_T_Peaks;

            if isempty(P_peaks)
                P_features = ["t_PR", "t_PQ", "t_PS", "t_PT", "t_PT_QS", "a_PQ", "a_PS", "a_PT", "a_PQ_QS", "a_PQ_QT", ...
                              "a_PQ_PS", "a_PQ_QR", "a_PQ_RS", "a_ST_PQ"];
                FEATURES_WAVES = rmfield(FEATURES_WAVES, P_features);
            end

            if isempty(Q_peaks)
                Q_features = ["t_QR", "t_PQ", "t_QS", "t_QT", "t_PT_QS", "t_QT_QS", "a_PQ", "a_QR", "a_QS", "a_QT", ...
                              "a_ST_QS", "a_RS_QR", "a_PQ_QS", "a_PQ_QT", "a_PQ_PS", "a_PQ_QR", "a_PQ_RS", ...
                              "a_RS_QS", "a_RS_QT", "a_ST_PQ", "a_ST_QT"];
                FEATURES_WAVES = rmfield(FEATURES_WAVES, Q_features);
            end

            if isempty(S_peaks)
                S_features = ["t_SR", "t_PS", "t_QS", "t_ST", "t_PT_QS", "t_QT_QS", "a_RS", "a_ST", "a_PS", "a_QS", ...
                              "a_ST_QS", "a_RS_QR", "a_PQ_QS", "a_PQ_PS", "a_PQ_RS", "a_RS_QS", "a_RS_QT", ...
                              "a_ST_PQ", "a_ST_QT"];
                FEATURES_WAVES = rmfield(FEATURES_WAVES, S_features);
            end

            if isempty(T_peaks)
                T_features = ["t_TR", "t_PT", "t_QT", "t_ST", "t_PT_QS", "t_QT_QS", "a_ST", "a_PT", "a_QT", "a_ST_QS", ...
                              "a_PQ_QT", "a_RS_QT", "a_ST_PQ", "a_ST_QT"];
                FEATURES_WAVES = rmfield(FEATURES_WAVES, T_features);
            end

            features_waves = struct();
            for m = 1:length(R_peaks)
                features = struct();
                keys = fieldnames(FEATURES_WAVES);
                for i = 1:length(keys)
                    key = keys{i};
                    func = FEATURES_WAVES.(key);
                    try
                        features.([obj.prefix, '_', key]) = func(sig, obj.sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, m);
                    catch
                        features.([obj.prefix, '_', key]) = NaN;
                    end
                end
                features_waves(m) = features;
            end

            if obj.average
                features_avr = struct();
                keys = fieldnames(features_waves(1));
                for i = 1:length(keys)
                    key = keys{i};
                    values = arrayfun(@(x) x.(key), features_waves, 'UniformOutput', false);
                    features_avr.(key) = mean(cell2mat(values));
                end
                features_waves = features_avr;
            end
        end
    end
end
