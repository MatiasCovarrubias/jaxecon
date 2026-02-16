function [peak_value, peak_period, half_life] = calculatePeaksAndHalfLives(ir_series)
% CALCULATEPEAKSANDHALFLIVES Calculate peak value, peak period, and half-life
%
% INPUTS:
%   ir_series - Vector of impulse response values (1 x T)
%
% OUTPUTS:
%   peak_value  - Maximum absolute deviation from steady state
%   peak_period - Period at which the peak occurs (0-indexed)
%   half_life   - Number of periods until IR decays to half of peak value
%
% Note: This function assumes the IR starts from steady state (0) and
%       tracks both when the peak occurs and when it decays to half.

    % Find the peak (maximum absolute value)
    [peak_value, peak_idx] = max(abs(ir_series));
    peak_period = peak_idx - 1;  % Convert to 0-indexed
    
    % Use the signed peak value
    peak_value = ir_series(peak_idx);
    
    % Calculate half of peak absolute value
    half_peak = abs(peak_value) / 2;
    
    % Find when the response crosses half of peak value after the peak
    half_life = NaN;
    for t = peak_idx:length(ir_series)
        if abs(ir_series(t)) <= half_peak
            half_life = t - peak_idx;
            break;
        end
    end
    
    % If never crosses, set to length of series
    if isnan(half_life)
        half_life = length(ir_series) - 1;
    end
end

