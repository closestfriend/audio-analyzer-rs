// =============================================================================
// report.rs — Machine-readable (JSON) full analysis
// =============================================================================
//
// The MCP server formats its results as text for an LLM to read. GUIs, scripts
// and batch tools need the same numbers as data. This module runs the same
// analysis pipeline as the `full_analysis` MCP tool and returns a
// `serde_json::Value` instead of a formatted string.
//
// Exposed on the CLI as:  cli --json <file> [--fps N] [--start S] [--end S]
//
// Every number here comes from the same library functions the MCP tool calls,
// so a GUI and Claude are always looking at identical measurements.

use serde_json::{Value, json};

use crate::analysis::{
    downsample, harmonic, masking, percussive, rhythm, sections, spectral, stereo, temporal,
};
use crate::{load_audio, load_audio_stereo};

/// Bump when the JSON shape changes in a way consumers must handle.
pub const SCHEMA_VERSION: u32 = 1;

const PITCH_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

pub struct ReportOptions {
    pub start_time: Option<f32>,
    pub end_time: Option<f32>,
    /// Time-series points per second. 0 disables time-series output.
    pub fps: f32,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            start_time: None,
            end_time: None,
            fps: 2.0,
        }
    }
}

/// Round to 4 decimals so f32→f64 conversion noise doesn't bloat the output.
/// Non-finite values become JSON null.
fn num(v: f32) -> Value {
    if v.is_finite() {
        json!((v as f64 * 10_000.0).round() / 10_000.0)
    } else {
        Value::Null
    }
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        f32::NAN
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn mean_columns<const N: usize>(frames: &[[f32; N]]) -> [f32; N] {
    let mut out = [0.0_f32; N];
    for frame in frames {
        for (i, &val) in frame.iter().enumerate() {
            out[i] += val;
        }
    }
    let n = frames.len().max(1) as f32;
    for val in &mut out {
        *val /= n;
    }
    out
}

/// Map the 7 band values to an object keyed by band name.
fn by_band(values: &[f32; 7]) -> Value {
    let mut map = serde_json::Map::new();
    for (i, &(name, _, _)) in spectral::FREQUENCY_BANDS.iter().enumerate() {
        map.insert(name.to_string(), num(values[i]));
    }
    Value::Object(map)
}

/// Highest frequency whose average magnitude is within `floor_db` of the
/// spectrum's peak. Lossy codecs lowpass the signal (e.g. ~16 kHz for 128 kbps
/// MP3), which distorts high-band metrics like brilliance contrast — consumers
/// can compare this against Nyquist to flag that.
fn effective_bandwidth_hz(spec: &spectral::Spectrogram, floor_db: f32) -> f32 {
    if spec.n_frames == 0 || spec.n_freq_bins == 0 {
        return f32::NAN;
    }
    let mut avg = vec![0.0_f32; spec.n_freq_bins];
    for frame in &spec.magnitudes {
        for (i, &m) in frame.iter().enumerate().take(spec.n_freq_bins) {
            avg[i] += m;
        }
    }
    let peak = avg.iter().cloned().fold(0.0_f32, f32::max).max(1e-12);
    let threshold = peak * 10.0_f32.powf(-floor_db / 20.0);
    let top_bin = avg.iter().rposition(|&m| m >= threshold).unwrap_or(0);
    spec.bin_to_freq(top_bin)
}

fn series(data: &[f32], native_fps: f32, fps: f32, offset: f32) -> (Vec<f32>, Vec<Value>) {
    let ds = downsample::downsample_f32(data, native_fps, fps);
    (
        ds.iter().map(|(t, _)| t + offset).collect(),
        ds.iter().map(|&(_, v)| num(v)).collect(),
    )
}

pub fn full_report(path: &str, opts: &ReportOptions) -> Result<Value, String> {
    let started = std::time::Instant::now();
    let mut audio = load_audio(path)?;
    let file_duration = audio.duration;

    // --- optional time slice (mirrors the MCP tool's start_time/end_time) ---
    let offset = opts.start_time.unwrap_or(0.0).max(0.0);
    let sr = audio.sample_rate;
    let start_sample = (offset * sr as f32) as usize;
    let end_sample = opts
        .end_time
        .map(|t| (t * sr as f32) as usize)
        .unwrap_or(audio.samples.len())
        .min(audio.samples.len());
    if start_sample >= end_sample {
        return Err(format!(
            "Invalid time range: {:.1}s–{:.1}s (file is {:.1}s)",
            offset,
            opts.end_time.unwrap_or(file_duration as f32),
            file_duration
        ));
    }
    let is_slice = opts.start_time.is_some() || opts.end_time.is_some();
    audio.samples = audio.samples[start_sample..end_sample].to_vec();
    audio.duration = audio.samples.len() as f64 / sr as f64;

    let spec = spectral::compute_spectrogram(&audio.samples, sr, None, None);

    // --- spectral / temporal ---
    let centroid = spectral::spectral_centroid(&spec);
    let bandwidth = spectral::spectral_bandwidth(&spec);
    let rolloff = spectral::spectral_rolloff(&spec, None);
    let flatness = spectral::spectral_flatness(&spec);
    let bands = spectral::frequency_band_energy(&spec);
    let contrast = spectral::spectral_contrast(&spec, None);
    let rms = temporal::rms_energy(&audio.samples, spec.n_fft, spec.hop_length);
    let zcr = temporal::zero_crossing_rate(&audio.samples, spec.n_fft, spec.hop_length);
    let mfccs = spectral::compute_mfccs(&spec, None, None);
    let n_mfcc = mfccs.first().map_or(0, |f| f.len());
    let mut avg_mfcc = vec![0.0_f32; n_mfcc];
    for frame in &mfccs {
        for (i, &v) in frame.iter().enumerate() {
            avg_mfcc[i] += v;
        }
    }
    for v in &mut avg_mfcc {
        *v /= mfccs.len().max(1) as f32;
    }

    let avg_bands = mean_columns(&bands.band_energies);
    // Upstream's "band energy" is RMS magnitude per FFT bin (an amplitude
    // density), so dB = 20*log10. Expressed relative to the loudest band so
    // tracks of different loudness are comparable.
    let loudest_band = avg_bands.iter().cloned().fold(0.0_f32, f32::max).max(1e-20);
    let mut band_level_db = [0.0_f32; 7];
    for (i, &e) in avg_bands.iter().enumerate() {
        band_level_db[i] = 20.0 * (e.max(1e-20) / loudest_band).log10();
    }
    let avg_contrast = mean_columns(&contrast.contrast);

    // --- harmonic ---
    let chromagram = harmonic::compute_chromagram(&audio.samples, sr, &spec);
    let (key, mode, key_confidence) = chromagram.estimate_key();
    let avg_chroma = mean_columns(&chromagram.chroma);
    let pitch_classes: Vec<Value> = PITCH_NAMES
        .iter()
        .zip(avg_chroma.iter())
        .map(|(name, &e)| json!({ "pitch": name, "energy": num(e) }))
        .collect();

    // --- rhythm / percussive ---
    let rhythm_result = rhythm::analyse_rhythm(&spec, None, None);
    let beat_stats = rhythm::beat_statistics(&rhythm_result.beat_times);
    let hpss = percussive::hpss(&spec, None);
    let perc = percussive::percussive_features(&hpss, sr, spec.hop_length);

    // --- masking (reuses HPSS) ---
    let h_spec = masking::spectrogram_from_hpss(&hpss.harmonic, sr, spec.n_fft, spec.hop_length);
    let p_spec = masking::spectrogram_from_hpss(&hpss.percussive, sr, spec.n_fft, spec.hop_length);
    let h_bands = spectral::frequency_band_energy(&h_spec);
    let p_bands = spectral::frequency_band_energy(&p_spec);
    let masking_result = masking::detect_masking(&bands, &contrast, &h_bands, &p_bands);
    let avg_crowding = mean_columns(&masking_result.analysis.crowding);
    let avg_collision = mean_columns(&masking_result.analysis.hp_collision);

    // --- dynamics / loudness / stereo ---
    let dr = temporal::dynamic_range(&audio.samples, spec.n_fft, spec.hop_length);
    let stereo_loaded = load_audio_stereo(path).ok().and_then(|s| {
        let end = end_sample.min(s.left.len());
        if start_sample >= end {
            None
        } else {
            Some((
                s.left[start_sample..end].to_vec(),
                s.right[start_sample..end].to_vec(),
                s.channels,
            ))
        }
    });
    let lufs = match stereo_loaded {
        Some((ref l, ref r, _)) => temporal::measure_lufs_stereo(l, r, sr),
        None => temporal::measure_lufs(&audio.samples, sr),
    };
    let stereo_analysis = stereo_loaded
        .as_ref()
        .map(|(l, r, ch)| stereo::analyse_stereo(l, r, *ch, spec.n_fft, spec.hop_length));
    let stereo_json = match stereo_analysis.as_ref() {
        Some(sa) => {
            let s = stereo::stereo_summary(sa);
            json!({
                "source_channels": sa.source_channels,
                "phase_correlation_avg": num(s.avg_phase_correlation),
                "phase_correlation_min": num(s.min_phase_correlation),
                "phase_warning_fraction": num(s.phase_warning_fraction),
                "width_avg": num(s.avg_stereo_width),
                "width_max": num(s.max_stereo_width),
                "balance_avg": num(s.avg_balance),
                "mono_compatibility_avg": num(s.avg_mono_compatibility),
                "mono_compatibility_min": num(s.min_mono_compatibility),
            })
        }
        None => Value::Null,
    };

    // --- sections (full-track only, as in the MCP tool) ---
    let sections_json = if is_slice {
        Value::Null
    } else {
        let s = sections::detect_sections(&spec, Some(&chromagram), None);
        json!({
            "working_bpm": num(s.working_bpm),
            "bpm_confidence": num(s.bpm_confidence),
            "boundaries": s.boundaries.iter().map(|b| json!({
                "time": num(b.time),
                "confidence": num(b.confidence),
                "signals": b.signals.iter().map(|sig| sig.label()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    };

    // --- time series ---
    let timeseries =
        if opts.fps > 0.0 {
            let native = downsample::native_fps(sr, spec.hop_length);
            let (time, rms_s) = series(&rms, native, opts.fps, offset);
            let mut ts = serde_json::Map::new();
            ts.insert("fps".into(), num(opts.fps));
            ts.insert(
                "time".into(),
                json!(time.iter().map(|&t| num(t)).collect::<Vec<_>>()),
            );
            ts.insert("rms".into(), json!(rms_s));
            ts.insert(
                "centroid_hz".into(),
                json!(series(&centroid, native, opts.fps, offset).1),
            );
            ts.insert(
                "crest_db".into(),
                json!(series(&dr.crest_factor_db, native, opts.fps, offset).1),
            );
            ts.insert(
                "onset".into(),
                json!(series(&rhythm_result.onset_envelope, native, opts.fps, offset).1),
            );
            ts.insert(
                "percussive_ratio".into(),
                json!(series(&perc.percussive_ratio, native, opts.fps, offset).1),
            );
            if let Some(sa) = stereo_analysis.as_ref() {
                ts.insert(
                    "phase_correlation".into(),
                    json!(series(&sa.phase_correlation, native, opts.fps, offset).1),
                );
                ts.insert(
                    "stereo_width".into(),
                    json!(series(&sa.stereo_width, native, opts.fps, offset).1),
                );
            }
            let ds_bands = downsample::downsample_array(&bands.band_energies, native, opts.fps);
            let mut band_series = serde_json::Map::new();
            for (i, &(name, _, _)) in spectral::FREQUENCY_BANDS.iter().enumerate() {
                band_series.insert(
                    name.to_string(),
                    json!(ds_bands.iter().map(|(_, a)| num(a[i])).collect::<Vec<_>>()),
                );
            }
            ts.insert("bands".into(), Value::Object(band_series));
            // Short-term LUFS has its own (3 s window) time base.
            ts.insert("lufs_short_term".into(), json!({
            "time": lufs.short_term_times.iter().map(|&t| num(t + offset)).collect::<Vec<_>>(),
            "value": lufs.short_term.iter().map(|&v| num(v)).collect::<Vec<_>>(),
        }));
            Value::Object(ts)
        } else {
            Value::Null
        };

    let bands_meta: Vec<Value> = spectral::FREQUENCY_BANDS
        .iter()
        .map(|&(name, lo, hi)| json!({ "name": name, "low_hz": lo, "high_hz": hi }))
        .collect();

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": env!("CARGO_PKG_VERSION"),
        "file": {
            "path": path,
            "sample_rate": sr,
            "duration": file_duration,
            "analysed_start": offset,
            "analysed_duration": audio.duration,
            "is_slice": is_slice,
            "nyquist_hz": sr as f32 / 2.0,
            "effective_bandwidth_hz": num(effective_bandwidth_hz(&spec, 60.0)),
        },
        "bands": bands_meta,
        "spectral": {
            "centroid_hz": num(mean(&centroid)),
            "bandwidth_hz": num(mean(&bandwidth)),
            "rolloff_hz": num(mean(&rolloff)),
            "flatness": num(mean(&flatness)),
            "rms": num(mean(&rms)),
            "zero_crossing_rate": num(mean(&zcr)),
            "mfcc": avg_mfcc.iter().map(|&v| num(v)).collect::<Vec<_>>(),
            "band_rms_magnitude": avg_bands.iter().map(|&v| json!(v as f64)).collect::<Vec<_>>(),
            "band_level_db": by_band(&band_level_db),
            "band_contrast_db": by_band(&avg_contrast),
        },
        "harmonic": {
            "key": key,
            "mode": mode,
            "key_confidence": num(key_confidence),
            "pitch_classes": pitch_classes,
        },
        "rhythm": {
            "tempo_bpm": num(rhythm_result.tempo_bpm),
            "tempo_confidence": num(rhythm_result.tempo_confidence),
            "beat_count": rhythm_result.beat_times.len(),
            "mean_bpm": beat_stats.as_ref().map(|s| num(s.mean_bpm)),
            "median_bpm": beat_stats.as_ref().map(|s| num(s.median_bpm)),
            "tempo_stability": beat_stats.as_ref().map(|s| num(s.tempo_stability)),
        },
        "percussive": {
            "percussive_ratio": num(mean(&perc.percussive_ratio)),
            "onset_density_per_sec": num(mean(&perc.onset_density)),
            "peak_attack_sharpness": num(perc.attack_sharpness.iter().cloned().fold(0.0_f32, f32::max)),
        },
        "dynamics": {
            "peak_dbfs": num(dr.peak_dbfs),
            "crest_factor_db": num(dr.overall_crest_db),
            "rms_range_db": num(dr.loudness_range_db),
            "rms_5th_db": num(dr.rms_5th_db),
            "rms_95th_db": num(dr.rms_95th_db),
        },
        "loudness": {
            "integrated_lufs": num(lufs.integrated),
            "true_peak_dbtp": num(lufs.true_peak_dbtp),
            "loudness_range_lu": num(lufs.loudness_range),
        },
        "masking": {
            "crowding": by_band(&avg_crowding),
            "hp_collision": by_band(&avg_collision),
            "adjacent_band_correlation": masking_result.cross_band.correlations.iter().map(|&v| num(v)).collect::<Vec<_>>(),
        },
        "stereo": stereo_json,
        "sections": sections_json,
        "timeseries": timeseries,
        "elapsed_sec": started.elapsed().as_secs_f32(),
    }))
}
