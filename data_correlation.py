
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Removed key, scale, chord_progression, spectral_peaks
#add them later, after making them into ints
connection = sqlite3.connect('sound_database.db')

df = pd.read_sql_query("""
    SELECT 
        yin_pitch_mean,
        yin_pitch_median,
        yin_pitch_std,
        yin_pitch_skewness,
        yin_pitch_kurtosis,
        yin_pitch_rms,
        yin_pitch_delta,
        melodia_pitch_mean,
        melodia_pitch_median,
        melodia_pitch_std,
        melodia_pitch_skewness,
        melodia_pitch_kurtosis,
        melodia_pitch_rms,
        melodia_pitch_delta,
        melodic_pitch_range,
        mnn_mean,
        mnn_median,
        mnn_std,
        mnn_skewness,
        mnn_kurtosis,
        mnn_rms,
        mnn_delta,
        inharmonicity_mean,
        inharmonicity_median,
        inharmonicity_std,
        inharmonicity_skewness,
        inharmonicity_kurtosis,
        inharmonicity_rms,
        inharmonicity_delta,
        chroma_mean,
        chroma_median,
        chroma_std,
        chroma_skewness,
        chroma_kurtosis,
        chroma_rms,
        chroma_delta,
        hpcp_mean,
        hpcp_median,
        hpcp_std,
        hpcp_skewness,
        hpcp_kurtosis,
        hpcp_rms,
        hpcp_delta,
        strength,
        dissonance,
        bpm_mean,
        bpm_median,
        bpm_std,
        bpm_skewness,
        bpm_kurtosis,
        bpm_delta,
        onset_rate,
        onset_mean,
        onset_median,
        onset_std,
        onset_skewness,
        onset_kurtosis,
        onset_rms,
        onset_delta,
        loudness_mean,
        loudness_median,
        loudness_std,
        loudness_skewness,
        loudness_kurtosis,
        loudness_rms,
        loudness_delta,
        dynamic_mean,
        dynamic_median,
        dynamic_std,
        dynamic_skewness,
        dynamic_kurtosis,
        dynamic_rms,
        dynamic_delta,
        rms_mean,
        rms_median,
        rms_std,
        rms_skewness,
        rms_kurtosis,
        rms_rms,
        rms_delta,
        mfcc_mean,
        mfcc_median,
        mfcc_std,
        mfcc_skewness,
        mfcc_kurtosis,
        mfcc_rms,
        mfcc_delta,
        centroid_mean,
        centroid_median,
        centroid_std,
        centroid_skewness,
        centroid_kurtosis,
        centroid_rms,
        centroid_delta,
        segment_mean,
        segment_median,
        segment_std,
        segment_skewness,
        segment_kurtosis,
        segment_rms,
        segment_delta,
        segment_count,
        novelty_mean,
        novelty_median,
        novelty_std,
        novelty_skewness,
        novelty_kurtosis,
        novelty_rms,
        novelty_delta,
        lat_mean,
        lat_median,
        lat_std,
        lat_skewness,
        lat_kurtosis,
        lat_rms,
        lat_delta,
        flatness_mean,
        flatness_median,
        flatness_std,
        flatness_skewness,
        flatness_kurtosis,
        flatness_rms,
        flatness_delta,
        t1_mean,
        t1_median,
        t1_std,
        t1_skewness,
        t1_kurtosis,
        t1_rms,
        t1_delta,
        t2_mean,
        t2_median,
        t2_std,
        t2_skewness,
        t2_kurtosis,
        t2_rms,
        t2_delta,
        t3_mean,
        t3_median,
        t3_std,
        t3_skewness,
        t3_kurtosis,
        t3_rms,
        t3_delta,
        harmonic_ratio_mean,
        harmonic_ratio_median,
        harmonic_ratio_std,
        harmonic_ratio_skewness,
        harmonic_ratio_kurtosis,
        harmonic_ratio_rms,
        harmonic_ratio_delta,
        danceability,
        dynamic_complexity
    FROM processed_sound_data
""", connection)

# Correlation threshold to filter out weak correlations
threshold = 0.75
correlation_matrix = df.corr()

#print(correlation_matrix)

for i in correlation_matrix.columns[1:]:
    for j in correlation_matrix.index:
        correlation_value = correlation_matrix.loc[j, i]
        if (correlation_value > threshold or correlation_value < -threshold) and correlation_value != 1:
            print(f"{j} vs {i}: {correlation_value:.4f}")
    

 
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm')
plt.title('Full Correlation Matrix')
plt.tight_layout()
plt.savefig('full_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

filtered_corr_df = correlation_matrix[((correlation_matrix > threshold) | (correlation_matrix < -threshold)) & (correlation_matrix != 1)] 

plt.figure(figsize=(10, 10))
sns.heatmap(filtered_corr_df, annot=False, cmap="Reds")
plt.title(f'Filtered Correlation Matrix (threshold > {threshold})')
plt.tight_layout()
plt.savefig('filtered_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Recommended features to keep (one from each highly correlated group)
features_to_keep = [
    # Pitch features - keep primary representatives
    'yin_pitch_mean',           # Keep mean over median/std/rms (most interpretable)
    'melodia_pitch_mean',       # Different algorithm, keep for comparison
    'melodic_pitch_range',      # Unique pitch span information
    
    # MNN features - highly correlated with melodia, keep just one
    'mnn_mean',                 # Keep as it's fundamental
    
    # Inharmonicity - keep mean as primary measure
    'inharmonicity_mean',       # Most interpretable
    
    # Chroma features - extremely high correlation, keep one
    'chroma_mean',              # Most fundamental representation
    
    # HPCP - different from chroma, keep mean
    'hpcp_mean',                # Harmonic pitch class profile
    
    # Rhythm features
    'bpm_mean',                 # Primary tempo measure
    'onset_rate',               # Rhythm density (unique information)
    
    # Loudness - many are highly correlated, keep primary
    'loudness_mean',            # Most interpretable loudness measure
    
    # Timbre features
    'mfcc_mean',                # Primary timbre descriptor
    'centroid_mean',            # Spectral brightness
    
    # Harmonic features
    'strength',                 # Harmonic strength
    'dissonance',               # Harmonic complexity
    
    # Spectral features
    'flatness_mean',            # Spectral flatness (noisiness)
    
    # Advanced features
    't1_mean',                  # Tristimulus 1 (fundamental vs harmonics)
    't2_mean',                  # Tristimulus 2 (low harmonics)
    't3_mean',                  # Tristimulus 3 (high harmonics)
    
    # Segment analysis
    'segment_count',            # Structural information
    'novelty_mean',             # Musical novelty/change detection
    
    # High-level features
    'danceability',             # Target-relevant feature
    'dynamic_complexity'        # Overall complexity measure
]

print(f"Recommended features to keep: {len(features_to_keep)}")
print("Features selected:")
for feature in sorted(features_to_keep):
    print(f"- {feature}")

# Create cleaned dataset
df_clean = df[features_to_keep]

# Verify no high correlations remain
clean_corr = df_clean.corr()
high_corr_remaining = []
for i in range(len(clean_corr.columns)):
    for j in range(i+1, len(clean_corr.columns)):
        corr_val = abs(clean_corr.iloc[i, j])
        if corr_val > 0.75:
            high_corr_remaining.append((clean_corr.columns[i], clean_corr.columns[j], corr_val))

print(f"\nRemaining high correlations (>0.75): {len(high_corr_remaining)}")
for feat1, feat2, corr_val in high_corr_remaining:
    print(f"- {feat1} vs {feat2}: {corr_val:.4f}")

# Save cleaned dataset
df_clean.to_csv('cleaned_features.csv', index=False)
print(f"\nDataset reduced from {len(df.columns)} to {len(df_clean.columns)} features")
print("Cleaned dataset saved to 'cleaned_features.csv'")

# Create correlation matrix for cleaned data
plt.figure(figsize=(12, 10))
sns.heatmap(clean_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Cleaned Dataset Correlation Matrix')
plt.tight_layout()
plt.savefig('cleaned_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Cleaned correlation heatmap saved to 'cleaned_correlation_heatmap.png'")

