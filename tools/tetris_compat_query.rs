//! Temporary read-only placement queries for the staged browser handover.
use super::*;
use std::collections::{BTreeSet, VecDeque};
pub const RULES: &str = "placement-v1-normalized-cw-no-kicks-7bag-query-feature-cost-v1";
const MAX_TURNS: u32 = 128;
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pose {
    pub r: u8,
    pub x: i8,
    pub y: i8,
}
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Candidate {
    pub pose: Pose,
    pub cells: Vec<(i8, i8)>,
    pub holes: u16,
    pub height: u8,
    pub lines: u8,
    pub evaluation: i32,
    pub path: Vec<Pose>,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct Receipt {
    pub version: u8,
    pub rules: String,
    pub id: u32,
    pub game: u32,
    pub seed: u32,
    pub mode: u8,
    pub turn: u32,
    pub piece: u8,
    pub next: u8,
    pub before: Vec<u8>,
    pub after: Vec<u8>,
    pub total_lines: u32,
    pub over: bool,
    pub legal_count: usize,
    pub candidates: Vec<Candidate>,
    pub recommended: usize,
    pub selected: usize,
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub model: Digest,
    pub prompt: String,
    pub input_tokens: u32,
    pub inference_instructions: String,
    pub timestamp_ns: String,
    pub previous_hash: String,
}
fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("bounded demo JSON")
}
fn shapes(piece: u8) -> Vec<Vec<(i8, i8)>> {
    let mut s = match piece {
        0 => vec![(0, 0), (1, 0), (2, 0), (3, 0)],
        1 => vec![(0, 0), (1, 0), (0, 1), (1, 1)],
        2 => vec![(0, 0), (1, 0), (2, 0), (1, 1)],
        3 => vec![(1, 0), (2, 0), (0, 1), (1, 1)],
        4 => vec![(0, 0), (1, 0), (1, 1), (2, 1)],
        5 => vec![(0, 0), (0, 1), (1, 1), (2, 1)],
        _ => vec![(2, 0), (0, 1), (1, 1), (2, 1)],
    };
    let mut result = vec![];
    for _ in 0..4 {
        let mx = s.iter().map(|c| c.0).min().unwrap();
        let my = s.iter().map(|c| c.1).min().unwrap();
        for c in &mut s {
            c.0 -= mx;
            c.1 -= my;
        }
        s.sort();
        if !result.contains(&s) {
            result.push(s.clone());
        }
        s = s.iter().map(|&(x, y)| (-y, x)).collect();
    }
    result
}
fn cells(s: &[Vec<(i8, i8)>], p: &Pose) -> Vec<(i8, i8)> {
    s[p.r as usize]
        .iter()
        .map(|&(x, y)| (x + p.x, y + p.y))
        .collect()
}
fn fits(board: &[u8], s: &[Vec<(i8, i8)>], p: &Pose) -> bool {
    cells(s, p).iter().all(|&(x, y)| {
        (0..10).contains(&x) && (0..20).contains(&y) && board[y as usize * 10 + x as usize] == 0
    })
}
pub fn landed(board: &[u8], piece: u8, cs: &[(i8, i8)]) -> (Vec<u8>, u8) {
    let mut b = board.to_vec();
    for &(x, y) in cs {
        b[y as usize * 10 + x as usize] = piece + 1;
    }
    let kept: Vec<u8> = b
        .chunks(10)
        .filter(|row| row.contains(&0))
        .flatten()
        .copied()
        .collect();
    let lines = ((200 - kept.len()) / 10) as u8;
    let mut after = vec![0; 200 - kept.len()];
    after.extend(kept);
    (after, lines)
}
fn features(b: &[u8]) -> (u16, u8) {
    let (mut holes, mut height) = (0, 0);
    for x in 0..10 {
        let mut covered = false;
        for y in 0..20 {
            if b[y * 10 + x] != 0 {
                covered = true;
                height = height.max(20 - y as u8);
            } else if covered {
                holes += 1;
            }
        }
    }
    (holes, height)
}
pub fn placements(board: &[u8], piece: u8) -> Vec<Candidate> {
    assert_eq!(board.len(), 200);
    let s = shapes(piece);
    let width = s[0].iter().map(|c| c.0).max().unwrap() + 1;
    let start = Pose {
        r: 0,
        x: (10 - width) / 2,
        y: 0,
    };
    if !fits(board, &s, &start) {
        return vec![];
    }
    let mut poses = vec![(start.clone(), None)];
    let mut queue = VecDeque::from([0]);
    let mut seen = BTreeSet::from([start]);
    let mut result = vec![];
    while let Some(i) = queue.pop_front() {
        let p = poses[i].0.clone();
        let down = Pose {
            y: p.y + 1,
            ..p.clone()
        };
        if !fits(board, &s, &down) {
            let cs = cells(&s, &p);
            let (b, lines) = landed(board, piece, &cs);
            let (holes, height) = features(&b);
            let mut path = vec![];
            let mut at = Some(i);
            while let Some(j) = at {
                path.push(poses[j].0.clone());
                at = poses[j].1;
            }
            path.reverse();
            result.push(Candidate {
                pose: p.clone(),
                cells: cs,
                holes,
                height,
                lines,
                evaluation: 10 * lines as i32 - 8 * holes as i32 - height as i32,
                path,
            });
        }
        for n in [
            Pose {
                r: (p.r + 1) % s.len() as u8,
                ..p.clone()
            },
            Pose {
                x: p.x - 1,
                ..p.clone()
            },
            Pose {
                x: p.x + 1,
                ..p.clone()
            },
            down,
        ] {
            if !seen.contains(&n) && fits(board, &s, &n) {
                seen.insert(n.clone());
                poses.push((n, Some(i)));
                queue.push_back(poses.len() - 1);
            }
        }
    }
    result.sort_by(|a, b| a.pose.cmp(&b.pose));
    result
}
fn sample_candidates(all: &[Candidate], limit: usize) -> Vec<Candidate> {
    let mut seen=BTreeSet::new();
    let unique:Vec<Candidate>=all.iter().filter(|c|seen.insert((c.lines,c.holes,c.height))).cloned().collect();
    if unique.is_empty() {return vec![];}
    let mut selected=vec![0usize];
    while selected.len()<limit.min(unique.len()) {
        let mut best=None;
        let mut best_distance=0u32;
        for (i,c) in unique.iter().enumerate() {
            if selected.contains(&i) {continue;}
            let distance=selected.iter().map(|&j| {
                let previous=&unique[j];
                c.lines.abs_diff(previous.lines) as u32
                    +c.holes.abs_diff(previous.holes) as u32
                    +c.height.abs_diff(previous.height) as u32
            }).min().unwrap();
            if best.is_none() || distance>=best_distance {
                best=Some(i);best_distance=distance;
            }
        }
        selected.push(best.unwrap());
    }
    selected.into_iter().map(|i|unique[i].clone()).collect()
}
pub fn recommend(options: &[Candidate]) -> usize {
    options.iter().enumerate().fold(0, |best, (i, c)| {
        if c.evaluation > options[best].evaluation {
            i
        } else {
            best
        }
    })
}
pub fn piece_at(seed: u32, index: u32) -> u8 {
    let mut state = if seed == 0 { 0x9e3779b9 } else { seed };
    let mut bag = [0u8, 1, 2, 3, 4, 5, 6];
    for b in 0..=index / 7 {
        bag = [0, 1, 2, 3, 4, 5, 6];
        for i in (1..7).rev() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let j = ((state as u64 * (i + 1) as u64) >> 32) as usize;
            bag.swap(i, j);
        }
        if b == index / 7 {
            break;
        }
    }
    bag[(index % 7) as usize]
}
fn rank(value: u16, values: &[u16]) -> &'static str {
    let min=*values.iter().min().unwrap();
    let max=*values.iter().max().unwrap();
    if min==max {"equal"} else if value==min {"min"} else if value==max {"max"} else {"mid"}
}
pub fn request(options: &[Candidate]) -> DecideRequest {
    let holes:Vec<u16>=options.iter().map(|c|c.holes).collect();
    let heights:Vec<u16>=options.iter().map(|c|c.height as u16).collect();
    let max_lines=options.iter().map(|c|c.lines).max().unwrap();
    let missed:Vec<u16>=options.iter().map(|c|(max_lines-c.lines) as u16).collect();
    DecideRequest {state:"Min holes".into(),question:String::new(),abstention:false,temperature:1.0,
        options:options.iter().enumerate().map(|(i,c)|OptionSpec {id:i.to_string(),
            text:format!("{} holes,{} height,{} missed",rank(c.holes,&holes),
                rank(c.height as u16,&heights),rank((max_lines-c.lines) as u16,&missed))}).collect()}
}

/// Stateless query input. The browser owns this board; it is not certified game state.
#[derive(CandidType, Deserialize, Serialize, Clone)]
pub struct TetrisQueryInput {
    pub board: Vec<u8>,
    pub seed: u32,
    pub turn: u32,
    pub lines: u32,
    pub mode: u8,
}
fn query_options(input: &TetrisQueryInput) -> std::result::Result<(usize, Vec<Candidate>), String> {
    if input.board.len() != 200
        || input.board.iter().any(|v| *v > 7)
        || input.turn >= MAX_TURNS
        || input.lines > input.turn * 4
        || input.mode > 1
    {
        return Err("Invalid query board or turn".into());
    }
    let all = placements(&input.board, piece_at(input.seed, input.turn));
    let options = sample_candidates(&all, 4);
    if options.is_empty() {
        return Err("No legal placements".into());
    }
    Ok((all.len(), options))
}
#[ic_cdk::query]
fn tetris_placement_query(input: TetrisQueryInput) -> std::result::Result<String, String> {
    billing::query_only().map_err(|e| e.to_string())?;
    let (legal_count, options) = query_options(&input)?;
    let recommended = recommend(&options);
    let req = request(&options);
    let prompt = verdict_candle::render_prompt(
        "",
        &req.state,
        &req.options
            .iter()
            .map(|o| o.text.clone())
            .collect::<Vec<_>>(),
    );
    let reply = if input.mode == 0 {
        Some(decide_once(req, QUERY_BUDGET).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let selected = if let Some(r) = &reply {
        let i = r
            .selected
            .parse::<usize>()
            .map_err(|_| "Invalid selected ID")?;
        if i >= options.len() {
            return Err("Invalid model selection".into());
        }
        i
    } else {
        recommended
    };
    let piece = piece_at(input.seed, input.turn);
    let next = piece_at(input.seed, input.turn + 1);
    let (after, lines) = landed(&input.board, piece, &options[selected].cells);
    let over = input.turn + 1 >= MAX_TURNS || placements(&after, next).is_empty();
    let receipt = Receipt {
        version: 2,
        rules: RULES.into(),
        id: input.turn + 1,
        game: 0,
        seed: input.seed,
        mode: input.mode,
        turn: input.turn,
        piece,
        next,
        before: input.board,
        after,
        total_lines: input.lines + lines as u32,
        over,
        legal_count,
        candidates: options,
        recommended,
        selected,
        logits: reply.as_ref().map(|r| r.logits.clone()).unwrap_or_default(),
        scores: reply
            .as_ref()
            .map(|r| r.probabilities.clone())
            .unwrap_or_default(),
        model: read(|s| s.active_model),
        prompt: if reply.is_some() {
            prompt
        } else {
            String::new()
        },
        input_tokens: reply.as_ref().map(|r| r.input_tokens).unwrap_or(0),
        inference_instructions: reply
            .as_ref()
            .map(|r| r.measured_instructions)
            .unwrap_or(0)
            .to_string(),
        timestamp_ns: ic_cdk::api::time().to_string(),
        previous_hash: String::new(),
    };
    let ids = (0..receipt.candidates.len()).collect::<Vec<_>>();
    let mut value = serde_json::to_value(receipt).expect("bounded query JSON");
    value["model_candidates"] = serde_json::json!(ids);
    Ok(json(&value))
}

#[ic_cdk::query]
fn tetris_demo_status() -> String {
    json(&serde_json::json!({"enabled": true, "warmed": MODEL.with(|m| m.borrow().is_some()),
        "model_turns_left": 0, "games_left": 0, "candidate_limit": 4,
        "turn_limit": MAX_TURNS, "rules": RULES}))
}
