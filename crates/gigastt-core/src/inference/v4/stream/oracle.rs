//! Opt-in frozen streaming differential test. Only private side-loaded assets.
use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

fn num(v: &Value) -> AnyResult<usize> {
    Ok(usize::try_from(v.as_u64().context("missing integer")?)?)
}
fn string(v: &Value) -> AnyResult<&str> {
    v.as_str().context("missing string")
}
fn array(v: &Value) -> AnyResult<&Vec<Value>> {
    v.as_array().context("missing array")
}
fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(2 * bytes.len());
    for byte in bytes {
        write!(&mut result, "{byte:02x}").expect("string write");
    }
    result
}
fn digest(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes).as_slice())
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn float_values(bytes: &[u8]) -> AnyResult<Vec<f32>> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    ensure!(remainder.is_empty(), "unaligned f32 data");
    let values = chunks
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite f32 data");
    Ok(values)
}
fn read_json(path: &Value, expected: &Value) -> AnyResult<Value> {
    let b = fs::read(string(path)?)?;
    ensure!(digest(&b) == string(expected)?, "JSON SHA mismatch");
    Ok(serde_json::from_slice(&b)?)
}
fn write_json(path: &Path, value: &Value) -> AnyResult<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn dims(v: &Value) -> AnyResult<Vec<usize>> {
    array(v)?.iter().map(num).collect()
}
fn metric(a: &[f32], b: &[f32], kind: &str) -> Value {
    if a.len() != b.len() {
        return json!({"pass":false,"actual":a.len(),"reference":b.len()});
    }
    let mut max = 0.0_f64;
    let mut sum = 0.0;
    let mut n = 0usize;
    let mut active_max = 0.0_f64;
    let mut active_sum = 0.0;
    let mut violations = 0;
    for (&a, &b) in a.iter().zip(b) {
        let diff = (f64::from(a) - f64::from(b)).abs();
        max = max.max(diff);
        sum += diff;
        if diff > 0.002 + 0.002 * f64::from(b).abs() {
            violations += 1;
        }
        if (f64::from(b) - f64::from(1.0e-9_f32.ln())).abs() > 0.05 {
            n += 1;
            active_sum += diff;
            active_max = active_max.max(diff);
        }
    }
    let mean = if a.is_empty() {
        0.0
    } else {
        sum / a.len() as f64
    };
    let active_mean = if n == 0 { 0.0 } else { active_sum / n as f64 };
    let exact = a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
    let pass = match kind {
        "exact" => exact,
        "mel" => max < 0.001 && active_max < 0.001 && active_mean < 0.00001,
        _ => violations == 0 && mean <= 0.001,
    };
    json!({"pass":pass,"count":a.len(),"max_abs":max,"mean_abs":mean,"bit_exact":exact,"active_count":n,"active_max":active_max,"active_mean":active_mean,"element_violations":violations})
}
struct Archive {
    raw: File,
    offset: u64,
    source: File,
    index: Value,
    arrays: BTreeMap<String, Value>,
    checks: BTreeMap<String, Value>,
    pass: bool,
}
impl Archive {
    fn new(path: &Path, index: Value) -> AnyResult<Self> {
        Ok(Self {
            raw: File::create_new(path)?,
            offset: 0,
            source: File::open(string(&index["npz_path"])?)?,
            index,
            arrays: BTreeMap::new(),
            checks: BTreeMap::new(),
            pass: true,
        })
    }
    fn check(&mut self, key: &str, pass: bool) {
        self.pass &= pass;
        self.checks.insert(key.into(), json!({"pass":pass}));
    }
    fn compare(
        &mut self,
        key: &str,
        shape: &[usize],
        values: &[f32],
        kind: &str,
        cache_suffix: bool,
        zero_seed: bool,
    ) -> AnyResult<()> {
        ensure!(
            shape.iter().product::<usize>() == values.len(),
            "native archive shape mismatch: {key}"
        );
        let data = bytes(values);
        let size = data.len() as u64;
        self.raw.write_all(&data)?;
        self.arrays.insert(
            key.into(),
            json!({"offset":self.offset,"bytes":size,"shape":shape,"sha256":digest(&data)}),
        );
        self.offset += size;
        for lane in ["pt", "ort"] {
            let ref_key = format!("{lane}__{key}");
            let reference = &self.index["arrays"][&ref_key];
            let mut check = if reference.is_null() {
                json!({"pass":zero_seed&&values.iter().all(|v|v.to_bits()==0),"reference_zero_seed":zero_seed,"reference_missing":true})
            } else {
                ensure!(reference["dtype"] == "float32", "wrong ref dtype");
                let mut raw = vec![0; num(&reference["bytes"])?];
                self.source.seek(SeekFrom::Start(
                    reference["offset"].as_u64().context("offset")?,
                ))?;
                self.source.read_exact(&mut raw)?;
                ensure!(
                    digest(&raw) == string(&reference["sha256"])?,
                    "raw range SHA mismatch"
                );
                let expected = float_values(&raw)?;
                let ref_shape = dims(&reference["shape"])?;
                let view = if ref_shape == shape {
                    Some(values)
                } else if lane == "pt"
                    && cache_suffix
                    && shape.len() == 3
                    && ref_shape.len() == 3
                    && shape[0] == 1
                    && ref_shape[0] == 1
                    && shape[2] == ref_shape[2]
                    && ref_shape[1] <= shape[1]
                {
                    Some(&values[(shape[1] - ref_shape[1]) * shape[2]..])
                } else {
                    None
                };
                view.map_or_else(
                    || json!({"pass":false,"actual_shape":shape,"reference_shape":ref_shape}),
                    |view| metric(view, &expected, kind),
                )
            };
            // Author's physical padded encoder tail never reaches decoder and
            // is descriptive only, exactly as the prior helper qualification.
            let descriptive = key == "encoded_physical";
            if !descriptive {
                self.pass &= check["pass"] == true;
            } else {
                check["descriptive_only"] = json!(true);
            }
            self.checks.insert(ref_key, check);
        }
        Ok(())
    }
    fn tensor(&mut self, key: &str, tensor: &Tensor, suffix: bool) -> AnyResult<()> {
        self.compare(
            key,
            tensor.shape().dims(),
            tensor.view().data().as_f32().context("native cache type")?,
            "numeric",
            suffix,
            true,
        )
    }
}
fn token_ids(tokens: &[V4Token]) -> Vec<usize> {
    tokens.iter().map(|t| t.id).collect()
}
fn frames(tokens: &[V4Token]) -> Vec<usize> {
    tokens.iter().map(|t| t.frame).collect()
}
fn confidence(tokens: &[V4Token]) -> Vec<f64> {
    tokens.iter().map(|t| t.confidence).collect()
}
fn near_number(a: f64, b: &Value, tolerance: f64) -> bool {
    b.as_f64().is_some_and(|b| (a - b).abs() <= tolerance)
}
fn close_list(a: &[f64], b: &Value, tolerance: f64) -> bool {
    b.as_array().is_some_and(|b| {
        a.len() == b.len() && a.iter().zip(b).all(|(&a, b)| near_number(a, b, tolerance))
    })
}
fn expected_label(v: &Value) -> AnyResult<usize> {
    if v.is_null() { Ok(1024) } else { num(v) }
}
fn expected_prediction(v: &Value) -> AnyResult<Vec<[usize; 2]>> {
    array(v)?
        .iter()
        .map(|x| Ok([expected_label(&x["label"])?, num(&x["local_frame"])?]))
        .collect()
}
fn expected_joint(v: &Value) -> AnyResult<Vec<[usize; 4]>> {
    array(v)?
        .iter()
        .map(|x| {
            Ok([
                num(&x["local_frame"])?,
                expected_label(&x["label"])?,
                num(&x["repeat"])?,
                num(&x["valid_rows"])?,
            ])
        })
        .collect()
}
fn check_head(
    archive: &mut Archive,
    name: &str,
    delta: &V4StreamingDelta,
    state: &HeadState,
    metadata: &Value,
) -> AnyResult<Value> {
    let ids = token_ids(&delta.tokens);
    let token_frames = frames(&delta.tokens);
    let probs = confidence(&delta.tokens);
    let words=delta.words.iter().map(|w|json!({"text":w.word,"start":w.start,"end":w.end,"confidence":w.confidence,"speaker":w.speaker})).collect::<Vec<_>>();
    let trace = &delta.trace;
    let state_metadata = json!({"previous":if state.greedy.label==1024{Value::Null}else{json!(state.greedy.label)},"frame_offset":state.greedy.frame_offset,"tokens":token_ids(&state.tokens),"frames":frames(&state.tokens),"published":state.published,"published_words":state.published_words,"prediction_some":state.greedy.prediction.is_some()});
    for lane in ["pt", "ort"] {
        let expected = &metadata[lane]["replay"][name];
        let out = &metadata[lane]["result"][name];
        let after = &metadata[lane]["after"]["heads"][name];
        let confidence_groups =
            array(&archive.index["published_word_confidence_groups"][lane][name])?;
        let w = array(&out["words"])?;
        let word_ok = w.len() == words.len()
            && w.len() == confidence_groups.len()
            && words
                .iter()
                .zip(w.iter().zip(confidence_groups))
                .all(|(a, (b, g))| {
                    a["text"] == b["text"]
                        && a["text"] == g["text"]
                        && ["start", "end"]
                            .iter()
                            .all(|k| near_number(a[k].as_f64().unwrap_or(f64::NAN), &b[k], 1e-9))
                        && near_number(
                            a["confidence"].as_f64().unwrap_or(f64::NAN),
                            &g["confidence"],
                            0.001,
                        )
                        && a["speaker"].is_null()
                });
        archive.check(
            &format!("{lane}_{name}_publication"),
            out["text"] == delta.text && word_ok,
        );
        archive.check(
            &format!("{lane}_{name}_raw_trajectory"),
            expected["tokens"] == json!(ids)
                && expected["absolute_frames"] == json!(token_frames)
                && close_list(&probs, &expected["confidence"], 0.001)
                && trace.predictor_inputs == expected_prediction(&expected["predictor_inputs"])?
                && trace.joint_inputs == expected_joint(&expected["joint_inputs"])?
                && expected["predictor_calls"] == json!(trace.predictor_inputs.len())
                && expected["joiner_calls"] == json!(trace.joint_inputs.len()),
        );
        archive.check(
            &format!("{lane}_{name}_state"),
            [
                "previous",
                "frame_offset",
                "tokens",
                "frames",
                "published",
                "published_words",
            ]
            .iter()
            .all(|&k| state_metadata[k] == after[k])
                && state.greedy.prediction.is_some() == !after["state"]["prediction"].is_null(),
        );
    }
    for (i, ((pred, h), c)) in trace
        .prediction
        .as_chunks::<320>()
        .0
        .iter()
        .zip(trace.h.as_chunks::<320>().0.iter())
        .zip(trace.c.as_chunks::<320>().0.iter())
        .enumerate()
    {
        for (key, values) in [("prediction", pred), ("h", h), ("c", c)] {
            archive.compare(
                &format!("{name}_p{}_{key}", i + 1),
                &[1, 1, 320],
                values,
                "numeric",
                false,
                false,
            )?;
        }
    }
    for (i, logp) in trace
        .log_probs
        .as_chunks::<{ 8 * 1025 }>()
        .0
        .iter()
        .enumerate()
    {
        archive.compare(
            &format!("{name}_j{}_log_probs", i + 1),
            &[1, 8, 1, 1025],
            logp,
            "numeric",
            false,
            false,
        )?;
    }
    archive.compare(
        &format!("{name}_state_hidden_h"),
        &[1, 1, 320],
        &state.greedy.h,
        "numeric",
        false,
        true,
    )?;
    archive.compare(
        &format!("{name}_state_hidden_c"),
        &[1, 1, 320],
        &state.greedy.c,
        "numeric",
        false,
        true,
    )?;
    if let Some(p) = &state.greedy.prediction {
        for (key, values) in [
            ("prediction", &p.values),
            ("next_h", &p.h),
            ("next_c", &p.c),
        ] {
            archive.compare(
                &format!("{name}_state_{key}"),
                &[1, 1, 320],
                values,
                "numeric",
                false,
                false,
            )?;
        }
    }
    Ok(
        json!({"text":delta.text,"words":words,"tokens":ids,"frames":token_frames,"confidence":probs,"state":state_metadata,"predictor_inputs":trace.predictor_inputs,"joint_inputs":trace.joint_inputs}),
    )
}

#[derive(Default)]
struct Counts {
    api: usize,
    feature: usize,
    front: usize,
    valid: usize,
    physical: usize,
    heads: usize,
    predict: usize,
    joint: usize,
    noop: usize,
    trace_bytes: u64,
}
impl Counts {
    fn record(&mut self, p: &V4StreamingPacket, bytes: u64) {
        self.api += 1;
        self.feature += usize::from(!p.trace.new_mel.is_empty());
        self.front += usize::from(p.trace.valid_encoded > 0);
        self.valid += p.trace.valid_encoded;
        self.physical += p.trace.physical_encoded;
        for h in [&p.e2e, &p.raw].into_iter().flatten() {
            self.heads += usize::from(p.trace.valid_encoded > 0);
            self.predict += h.trace.predictor_inputs.len();
            self.joint += h.trace.joint_inputs.len();
        }
        self.trace_bytes += bytes;
    }
    fn value(&self) -> Value {
        json!({"api":self.api,"preprocessor":self.feature,"frontend":self.front,"step":self.front,"valid_encoded":self.valid,"physical_encoded":self.physical,"head_groups":self.heads,"predictors":self.predict,"joiners":self.joint,"scored_rows":self.joint*8,"noops":self.noop,"trace_bytes":self.trace_bytes})
    }
    fn bound(&self) -> AnyResult<()> {
        ensure!(
            self.api <= 1187
                && self.front <= 557
                && self.heads <= 1078
                && self.predict <= 91410
                && self.joint <= 91410
                && self.trace_bytes <= 8 * 1024 * 1024 * 1024,
            "frozen call/trace cap exceeded"
        );
        Ok(())
    }
}
struct PacketEvidence<'a> {
    pcm: &'a [f32],
    metadata: &'a Value,
    index: Value,
    physical: [usize; 7],
    cache_before: Option<String>,
}
fn evaluate_packet(
    session: &V4StreamingSession,
    packet: &V4StreamingPacket,
    out: &Path,
    counts: &mut Counts,
    evidence: PacketEvidence<'_>,
) -> AnyResult<Value> {
    let PacketEvidence {
        pcm,
        metadata,
        index,
        physical,
        cache_before,
    } = evidence;
    let state = session
        .state
        .as_ref()
        .context("missing native state after successful packet")?;
    let row = &metadata["row"];
    let trace = &packet.trace;
    let mut a = Archive::new(out, index)?;
    a.check(
        "geometry",
        packet.samples == num(&row["utterance_samples_after"])?
            && packet.finished == row["final"].as_bool().context("final")?
            && packet.encoded_frames == num(&row["position_after"])?
            && state.cache_valid == num(&row["export_cache_valid_after"])?
            && state.wave.len() == num(&row["wave_buffer_after_samples"])?
            && state.pending.len() == MELS * num(&row["mel_pending_after"])?
            && trace.new_mel.len() == MELS * num(&row["new_mel_frames"])?
            && trace.physical_mel == num(&row["ready_mel_physical"])?
            && trace.valid_mel == num(&row["ready_mel_valid"])?
            && trace.physical_encoded == num(&row["physical_encoded_frames"])?
            && trace.valid_encoded == num(&row["valid_encoded_frames"])?,
    );
    a.check(
        "head_selection",
        packet.raw.is_some() == session.styles.raw()
            && packet.e2e.is_some() == session.styles.e2e(),
    );
    let high = &metadata["ort"]["calls"];
    let expected = [
        "preprocessor",
        "frontend",
        "step",
        "e2e_predict",
        "e2e_joint",
        "raw_predict",
        "raw_joint",
    ]
    .map(|key| {
        if (key.starts_with("raw_") && !session.styles.raw())
            || (key.starts_with("e2e_") && !session.styles.e2e())
        {
            0
        } else {
            high[key].as_u64().unwrap_or(0) as usize
        }
    });
    a.check("physical_call_counts", physical == expected);
    a.compare("packet_waveform", &[pcm.len()], pcm, "exact", false, false)?;
    if !trace.feature_input.is_empty() {
        a.compare(
            "feature_input",
            &[1, trace.feature_input.len()],
            &trace.feature_input,
            "exact",
            false,
            false,
        )?;
    }
    a.compare(
        "new_mel",
        &[1, MELS, trace.new_mel.len() / MELS],
        &trace.new_mel,
        "mel",
        false,
        false,
    )?;
    a.compare(
        "ready_mel",
        &[1, MELS, trace.valid_mel],
        &trace.ready_mel,
        "mel",
        false,
        false,
    )?;
    if trace.valid_encoded > 0 {
        a.compare(
            "front_input",
            &[1, MELS, trace.physical_mel],
            &trace.padded_mel,
            "mel",
            false,
            false,
        )?;
        a.compare(
            "pre_encoded",
            &[1, trace.physical_encoded, CHANNELS],
            &trace.pre_encoded,
            "numeric",
            false,
            false,
        )?;
        a.compare(
            "encoded_physical",
            &[1, CHANNELS, trace.physical_encoded],
            &trace.encoded_physical,
            "numeric",
            false,
            false,
        )?;
    }
    a.compare(
        "encoded_valid",
        &[1, CHANNELS, trace.valid_encoded],
        &trace.encoded_valid,
        "numeric",
        false,
        false,
    )?;
    a.compare(
        "wave_after",
        &[state.wave.len()],
        &state.wave,
        "exact",
        false,
        false,
    )?;
    a.compare(
        "pending_after",
        &[1, MELS, state.pending.len() / MELS],
        &state.pending,
        "mel",
        false,
        false,
    )?;
    let cache_after = if let Some(before) = &cache_before {
        // The official archives only retain raw acoustic caches on admitted
        // rows. Check exact no-ready retention from both actual state
        // fingerprints and the official before/after descriptors instead.
        let after = fingerprint_caches(state);
        let reference_retained = ["pt", "ort"].into_iter().all(|lane| {
            ["front", "caches"]
                .into_iter()
                .all(|key| metadata[lane]["before"][key] == metadata[lane]["after"][key])
        });
        a.check(
            "no_ready_cache_unchanged",
            reference_retained && before == &after && trace.valid_encoded == 0,
        );
        Some(after)
    } else {
        for (i, t) in state.front.iter().enumerate() {
            a.tensor(&format!("front_state_{i}"), t, false)?;
        }
        for (i, t) in state.layers.iter().enumerate() {
            a.tensor(&format!("cache_state_{i}"), t, true)?;
        }
        None
    };
    let raw = packet
        .raw
        .as_ref()
        .zip(state.raw.as_ref())
        .map(|(delta, head)| check_head(&mut a, "raw", delta, head, metadata))
        .transpose()?;
    let e2e = packet
        .e2e
        .as_ref()
        .zip(state.e2e.as_ref())
        .map(|(delta, head)| check_head(&mut a, "e2e", delta, head, metadata))
        .transpose()?;
    a.raw.flush()?;
    counts.record(packet, a.offset);
    counts.bound()?;
    Ok(
        json!({"pass":a.pass,"arrays":a.arrays,"checks":a.checks,"raw":raw,"e2e":e2e,"samples":packet.samples,"encoded_frames":packet.encoded_frames,"cache_valid":state.cache_valid,"physical_calls":physical,"cache_before":cache_before,"cache_after":cache_after,"finished":packet.finished,"trace_bytes":a.offset,"row":row}),
    )
}
fn state_zero(session: &V4StreamingSession) -> bool {
    session.state.as_ref().is_some_and(|s| {
        s.samples == 0
            && !s.finished
            && s.position == 0
            && s.cache_valid == 0
            && s.pending.is_empty()
            && s.wave.is_empty()
            && s.front.iter().chain(&s.layers).all(|t| {
                t.view()
                    .data()
                    .as_f32()
                    .is_some_and(|d| d.iter().all(|v| v.to_bits() == 0))
            })
            && s.raw.iter().chain(&s.e2e).all(|h| {
                h.tokens.is_empty()
                    && h.published.is_empty()
                    && h.published_words == 0
                    && h.greedy.label == 1024
                    && h.greedy.frame_offset == 0
                    && h.greedy
                        .h
                        .iter()
                        .chain(&h.greedy.c)
                        .all(|v| v.to_bits() == 0)
                    && h.greedy.prediction.is_none()
            })
    })
}
fn fingerprint_caches(state: &State) -> String {
    let mut hash = Sha256::new();
    for t in state.front.iter().chain(&state.layers) {
        hash.update(t.shape().dims().len().to_le_bytes());
        for dim in t.shape().dims() {
            hash.update(dim.to_le_bytes());
        }
        hash.update(bytes(t.view().data().as_f32().unwrap_or_default()));
    }
    hex_digest(hash.finalize().as_slice())
}
fn fingerprint_state(state: &State) -> String {
    let mut hash = Sha256::new();
    for n in [
        state.samples,
        state.position,
        state.cache_valid,
        usize::from(state.finished),
    ] {
        hash.update(n.to_le_bytes());
    }
    hash.update(bytes(&state.wave));
    hash.update(bytes(&state.pending));
    for t in state.front.iter().chain(&state.layers) {
        hash.update(bytes(t.view().data().as_f32().unwrap_or_default()));
    }
    for h in state.raw.iter().chain(&state.e2e) {
        hash.update(h.greedy.label.to_le_bytes());
        hash.update(h.greedy.frame_offset.to_le_bytes());
        hash.update(bytes(&h.greedy.h));
        hash.update(bytes(&h.greedy.c));
        if let Some(p) = &h.greedy.prediction {
            hash.update(bytes(&p.values));
            hash.update(bytes(&p.h));
            hash.update(bytes(&p.c));
        }
        for t in &h.tokens {
            hash.update(t.id.to_le_bytes());
            hash.update(t.frame.to_le_bytes());
            hash.update(t.confidence.to_le_bytes());
        }
        hash.update(h.published.as_bytes());
        hash.update(h.published_words.to_le_bytes());
    }
    hex_digest(hash.finalize().as_slice())
}
struct CaseRun {
    case: Value,
    pcm: Vec<f32>,
    out: PathBuf,
    session: V4StreamingSession,
    next: usize,
    pass: bool,
    summaries: Vec<Value>,
}
impl CaseRun {
    fn new(
        case: &Value,
        engine: &Arc<V4StreamingEngine>,
        styles: V4StreamingStyles,
        out: &Path,
    ) -> AnyResult<Self> {
        let id = string(&case["id"])?;
        ensure!(
            id.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "invalid case id"
        );
        let pcm = case_pcm(case)?;
        let context = case_context(case)?;
        Self::with_session(case, pcm, engine.start_session(context, styles)?, out)
    }
    fn with_session(
        case: &Value,
        pcm: Vec<f32>,
        session: V4StreamingSession,
        out: &Path,
    ) -> AnyResult<Self> {
        fs::create_dir(out)?;
        let pass = state_zero(&session);
        write_json(
            &out.join("reset.json"),
            &json!({"pass":pass,"state_sha256":fingerprint_state(session.state.as_ref().context("reset state")?)}),
        )?;
        Ok(Self {
            case: case.clone(),
            pcm,
            out: out.to_owned(),
            session,
            next: 0,
            pass,
            summaries: Vec::new(),
        })
    }
    fn has_packet(&self) -> bool {
        self.case["packets"]
            .as_array()
            .is_some_and(|v| self.next < v.len())
    }
    fn step(&mut self, counts: &mut Counts, physical: &PhysicalCounts) -> AnyResult<()> {
        let p = &self.case["packets"][self.next];
        ensure!(num(&p["row_index"])? == self.next + 1, "packet sequence");
        let metadata = read_json(&p["packet_metadata_path"], &p["packet_metadata_sha256"])?;
        let index = read_json(&p["arrays_index_path"], &p["arrays_index_sha256"])?;
        let row = &metadata["row"];
        let pcm = &self.pcm[num(&row["source_start_sample"])?..num(&row["source_end_sample"])?];
        ensure!(
            digest(&bytes(pcm)) == string(&row["pcm_sha256"])?,
            "packet SHA"
        );
        let key = format!("packet-{:04}", self.next + 1);
        let finished = row["final"].as_bool().context("final flag")?;
        let before = physical.snapshot();
        let cache_before = if num(&row["valid_encoded_frames"])? == 0 {
            Some(fingerprint_caches(
                self.session.state.as_ref().context("cache before packet")?,
            ))
        } else {
            None
        };
        let result = if row["kind"] == "flush" {
            self.session.flush(None)
        } else {
            self.session.push_pcm(pcm, finished, None)
        };
        let actual = physical.delta(before);
        let summary_result = match result {
            Ok(packet) => evaluate_packet(
                &self.session,
                &packet,
                &self.out.join(format!("{key}.f32le")),
                counts,
                PacketEvidence {
                    pcm,
                    metadata: &metadata,
                    index,
                    physical: actual,
                    cache_before,
                },
            ),
            Err(error) => Err(error.into()),
        };
        let summary = match summary_result {
            Ok(s) => s,
            Err(error) => {
                write_json(
                    &self.out.join(format!("{key}-technical-error.json")),
                    &json!({"error":format!("{error:#}"),"row":row,"actual_calls":actual,"terminal_failed":self.session.state.is_none()}),
                )?;
                return Err(error);
            }
        };
        self.pass &= summary["pass"] == true;
        write_json(&self.out.join(format!("{key}.json")), &summary)?;
        self.summaries.push(json!({"pass":summary["pass"],"packet":key,"signature":digest(&serde_json::to_vec(&summary)?)}));
        self.next += 1;
        Ok(())
    }
    fn finish(&mut self, counts: &mut Counts, physical: &PhysicalCounts) -> AnyResult<Value> {
        let before = fingerprint_state(self.session.state.as_ref().context("finished state")?);
        for i in 1..=2 {
            let reference = &self.case["finished_noops"][i - 1];
            let reference = read_json(&reference["path"], &reference["sha256"])?;
            let noopspec = &reference["row"];
            let before_calls = physical.snapshot();
            let packet = if i == 1 {
                self.session.flush(None)?
            } else {
                let pcm = &self.pcm
                    [num(&noopspec["source_start_sample"])?..num(&noopspec["source_end_sample"])?];
                ensure!(
                    digest(&bytes(pcm)) == string(&noopspec["pcm_sha256"])?,
                    "finished noop PCM identity"
                );
                self.session.push_pcm(pcm, false, None)?
            };
            let actual_calls = physical.delta(before_calls);
            let after =
                fingerprint_state(self.session.state.as_ref().context("post-finished state")?);
            let pass = reference["pass"] == true
                && actual_calls == [0; 7]
                && before == after
                && packet.finished
                && packet.samples == self.pcm.len()
                && [&packet.raw, &packet.e2e].into_iter().flatten().all(|h| {
                    h.text.is_empty()
                        && h.words.is_empty()
                        && h.tokens.is_empty()
                        && h.trace.predictor_inputs.is_empty()
                        && h.trace.joint_inputs.is_empty()
                })
                && packet.trace.new_mel.is_empty();
            self.pass &= pass;
            counts.api += 1;
            counts.noop += 1;
            counts.bound()?;
            write_json(
                &self.out.join(format!("finished-noop-{i}.json")),
                &json!({"pass":pass,"before":before,"after":after,"actual_calls":actual_calls}),
            )?;
        }
        let state = self.session.state.as_ref().context("final state")?;
        let prev = read_json(&self.case["result"]["path"], &self.case["result"]["sha256"])?;
        for (name, head) in [("raw", &state.raw), ("e2e", &state.e2e)] {
            if let Some(head) = head {
                for lane in ["pt", "ort"] {
                    self.pass &= prev["accumulated"][lane][name]["text"] == head.published
                        && close_list(
                            &confidence(&head.tokens),
                            &prev["accumulated"][lane][name]["token_confidence"],
                            0.001,
                        );
                }
            }
        }
        let summary = json!({"pass":self.pass,"case":self.case["id"],"packets":self.summaries,"final_fingerprint":before,"final_samples":state.samples,"final_position":state.position,"raw_text":state.raw.as_ref().map(|h|h.published.as_str()),"e2e_text":state.e2e.as_ref().map(|h|h.published.as_str())});
        write_json(&self.out.join("result.json"), &summary)?;
        Ok(summary)
    }
}
fn case_context(case: &Value) -> AnyResult<V4StreamContext> {
    V4StreamContext::from_millis(u16::try_from(num(&case["future_context_ms"])?)?)
}
fn case_pcm(case: &Value) -> AnyResult<Vec<f32>> {
    let data = fs::read(string(&case["pcm_path"])?)?;
    ensure!(digest(&data) == string(&case["pcm_sha256"])?, "PCM hash");
    let pcm = float_values(&data)?;
    ensure!(pcm.len() == num(&case["samples"])?, "PCM length");
    Ok(pcm)
}
#[derive(Default)]
struct PhysicalCounts([AtomicUsize; 7]);
impl PhysicalCounts {
    fn snapshot(&self) -> [usize; 7] {
        self.0.each_ref().map(|x| x.load(Ordering::Relaxed))
    }
    fn delta(&self, previous: [usize; 7]) -> [usize; 7] {
        let current = self.snapshot();
        std::array::from_fn(|i| current[i] - previous[i])
    }
}
struct CountedSession {
    session: Box<dyn RuntimeSession>,
    counts: Arc<PhysicalCounts>,
    index: usize,
}
impl RuntimeSession for CountedSession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, crate::runtime::RuntimeError> {
        self.counts.0[self.index].fetch_add(1, Ordering::Relaxed);
        self.session.run(inputs)
    }
}
fn instrument(engine: V4StreamingEngine, counts: &Arc<PhysicalCounts>) -> V4StreamingEngine {
    let V4StreamingEngine {
        mel,
        front,
        step,
        raw,
        e2e,
        request,
        active,
        pack_identity,
    } = engine;
    let wrap = |session, index| -> Box<dyn RuntimeSession> {
        Box::new(CountedSession {
            session,
            index,
            counts: counts.clone(),
        })
    };
    let acoustic = counts.clone();
    V4StreamingEngine {
        mel: Box::new(move |pcm| {
            acoustic.0[0].fetch_add(1, Ordering::Relaxed);
            mel(pcm)
        }),
        front: wrap(front, 1),
        step: wrap(step, 2),
        e2e: Head {
            tokenizer: e2e.tokenizer,
            predictor: wrap(e2e.predictor, 3),
            joiner: wrap(e2e.joiner, 4),
        },
        raw: Head {
            tokenizer: raw.tokenizer,
            predictor: wrap(raw.predictor, 5),
            joiner: wrap(raw.joiner, 6),
        },
        request,
        active,
        pack_identity,
    }
}
fn exact_counts(
    counts: &Counts,
    expected_api: usize,
    expected_noops: usize,
    expected_front: usize,
    expected_valid: usize,
    expected_heads: usize,
) -> bool {
    (
        counts.api,
        counts.noop,
        counts.front,
        counts.valid,
        counts.heads,
    ) == (
        expected_api,
        expected_noops,
        expected_front,
        expected_valid,
        expected_heads,
    )
}
fn case_output_name(case: &Value) -> AnyResult<&str> {
    let id = string(&case["id"])?;
    ensure!(
        !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
        "invalid case id"
    );
    Ok(id)
}

#[test]
#[ignore = "requires exact local streaming pack, qualified MKL runtime and frozen private oracle"]
fn test_v4_streaming_native_against_frozen_official_helpers() -> AnyResult<()> {
    let manifest = std::env::var("GIGASTT_V4_STREAM_ORACLE")?;
    let manifest_sha = std::env::var("GIGASTT_V4_STREAM_ORACLE_SHA256")?;
    let manifest = read_json(&json!(manifest), &json!(manifest_sha))?;
    let alternate = read_json(
        &json!(std::env::var("GIGASTT_V4_STREAM_ALTERNATE")?),
        &json!(std::env::var("GIGASTT_V4_STREAM_ALTERNATE_SHA256")?),
    )?;
    let output = PathBuf::from(std::env::var("GIGASTT_V4_STREAM_OUTPUT")?);
    let pack = PathBuf::from(std::env::var("GIGASTT_V4_STREAM_PACK")?);
    let mkl = PathBuf::from(std::env::var("GIGASTT_V4_STREAM_MKL")?);
    let cases = array(&manifest["cases"])?;
    ensure!(cases.len() == 19, "expected exact nineteen standard cases");
    let alt_cases = array(&alternate["cases"])?;
    ensure!(
        alt_cases.len() == 1,
        "expected one alternate partition oracle"
    );
    let alt = &alt_cases[0];
    ensure!(
        alt["source_id"] == "ru_short" && alt["future_context_ms"] == 320,
        "alternate source/profile"
    );
    fs::create_dir(&output)?;
    let counts = Arc::new(PhysicalCounts::default());
    let engine = match V4StreamingEngine::load(&pack, &mkl) {
        Ok(engine) => Arc::new(instrument(engine, &counts)),
        Err(e) => {
            write_json(
                &output.join("load-error.json"),
                &json!({"error":format!("{e:?}"),"actual_calls":counts.snapshot()}),
            )?;
            return Err(e.into());
        }
    };
    let mut actual = Counts::default();
    let mut summaries = BTreeMap::new();
    let mut pass = true;
    let result = (|| -> AnyResult<()> {
        let mut prior = None::<V4StreamingSession>;
        for case in cases {
            let id = case_output_name(case)?;
            let mut run = if let Some(mut session) = prior.take() {
                session.reset(case_context(case)?)?;
                CaseRun::with_session(case, case_pcm(case)?, session, &output.join(id))?
            } else {
                CaseRun::new(case, &engine, V4StreamingStyles::Both, &output.join(id))?
            };
            while run.has_packet() {
                run.step(&mut actual, &counts)?;
            }
            let mut summary = run.finish(&mut actual, &counts)?;
            if let Some(previous) = case["same_context_exact_reset_replay_of"].as_str() {
                let expected: &Value = summaries
                    .get(previous)
                    .context("missing reset replay reference")?;
                let exact = summary["packets"] == expected["packets"]
                    && summary["final_fingerprint"] == expected["final_fingerprint"];
                summary["same_context_reset_bit_exact"] = json!(exact);
                summary["pass"] = json!(summary["pass"] == true && exact);
                write_json(&run.out.join("result.json"), &summary)?;
            }
            pass &= summary["pass"] == true;
            summaries.insert(id.to_owned(), summary);
            prior = Some(run.session);
        }
        pass &= exact_counts(&actual, 1109, 38, 514, 3930, 1028)
            && actual.feature == 1037
            && actual.physical == 4028
            && actual.predict == 1241
            && actual.joint == 2395;
        write_json(
            &output.join("standard-counts.json"),
            &json!({"counts":actual.value(),"physical":counts.snapshot(),"pass":pass}),
        )?;
        drop(prior);
        let id = case_output_name(alt)?;
        let mut run = CaseRun::new(alt, &engine, V4StreamingStyles::Both, &output.join(id))?;
        while run.has_packet() {
            run.step(&mut actual, &counts)?;
        }
        let summary = run.finish(&mut actual, &counts)?;
        pass &= summary["pass"] == true;
        summaries.insert(id.to_owned(), summary);
        drop(run);
        pass &= exact_counts(&actual, 1118, 40, 521, 4044, 1042);
        write_json(
            &output.join("alternate-counts.json"),
            &json!({"counts":actual.value(),"physical":counts.snapshot(),"pass":pass}),
        )?;
        let a = cases
            .iter()
            .find(|c| c["id"] == "ctx160-ru_short")
            .context("missing interleave raw")?;
        let b = cases
            .iter()
            .find(|c| c["id"] == "ctx640-en_short")
            .context("missing interleave e2e")?;
        let mut a = CaseRun::new(
            a,
            &engine,
            V4StreamingStyles::Raw,
            &output.join("interleave-ctx160-ru_short-raw"),
        )?;
        let mut b = CaseRun::new(
            b,
            &engine,
            V4StreamingStyles::E2e,
            &output.join("interleave-ctx640-en_short-e2e"),
        )?;
        let owner_limit = engine
            .start_session(V4StreamContext::Ms320, V4StreamingStyles::Both)
            .is_err();
        pass &= owner_limit;
        while a.has_packet() || b.has_packet() {
            if a.has_packet() {
                a.step(&mut actual, &counts)?;
            }
            if b.has_packet() {
                b.step(&mut actual, &counts)?;
            }
        }
        for (id, run) in [
            ("interleave-ctx160-ru_short-raw", &mut a),
            ("interleave-ctx640-en_short-e2e", &mut b),
        ] {
            let done = run.finish(&mut actual, &counts)?;
            pass &= done["pass"] == true;
            summaries.insert(id.to_owned(), done);
        }
        drop(a);
        drop(b);
        let ownership_released =
            engine.active.load(Ordering::Acquire) == 0 && engine.request.try_lock().is_some();
        pass &= ownership_released;
        write_json(
            &output.join("slot-release.json"),
            &json!({"third_state_rejected":owner_limit,"both_released":ownership_released,"pass":owner_limit&&ownership_released}),
        )?;
        pass &= exact_counts(&actual, 1187, 44, 557, 4266, 1078)
            && actual.feature == 1105
            && actual.physical == 4376
            && counts.snapshot() == [1105, 557, 557, 706, 1302, 614, 1231];
        // Every physical native call is observed by a delegating wrapper; no
        // inferred call count is used as evidence of absent extra execution.
        let physical = counts.snapshot();
        pass &= (
            physical[0],
            physical[1],
            physical[2],
            physical[3] + physical[5],
            physical[4] + physical[6],
        ) == (
            actual.feature,
            actual.front,
            actual.front,
            actual.predict,
            actual.joint,
        );
        Ok(())
    })();
    let failure = result.as_ref().err().map(|e| format!("{e:#}"));
    write_json(
        &output.join("completion.json"),
        &json!({"pass":pass&&result.is_ok(),"failure":failure,"pack_identity":engine.pack_identity(),"counts":actual.value(),"physical_counts":counts.snapshot(),"cases":summaries}),
    )?;
    result?;
    ensure!(
        pass,
        "one or more frozen native streaming parity checks failed"
    );
    Ok(())
}
