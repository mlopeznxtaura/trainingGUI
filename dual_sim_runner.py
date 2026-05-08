"""
dual_sim_runner.py — Dual-repo simulation harness

Runs sentient-edge-node (perception/decision) AND trainingGUI (inference/adapter)
simultaneously, pairing their outputs into unified JSONL records.

Each record:
{
  "iter": int,
  "timestamp_ms": float,
  "repo": "dual",
  "edge": { frame, detections, depth, audio, decision, actuator_outputs },
  "inference": { framework, adapter, output, latency_ms, weight_state },
  "cross": { decision_matches_inference, latency_delta_ms, agreement_score },
  "memory": { short_term_count, long_term, decisions_made },
  "session_id": str
}
"""
import time, json, random, logging, os, sys, uuid, math
from dataclasses import asdict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger("dual_sim")

OUTPUT_PATH = os.environ.get("SIM_OUTPUT", "/opt/data/sim/dual_iterations.jsonl")
MAX_ITERS   = int(os.environ.get("SIM_ITERS", "100"))
SESSION_ID  = str(uuid.uuid4())[:8]

CLASSES    = ["person", "car", "bicycle", "dog", "laptop", "chair", "bottle", "phone"]
ZONES      = ["entry", "restricted", "public", "perimeter"]
FRAMEWORKS = ["pytorch", "llamacpp", "llm_provider", "onnx", "tensorflow"]
ACTIONS    = ["alert", "track", "ignore", "log", "speak"]

# ── Synthetic edge-node perception ─────────────────────────────────────────────

def synthetic_frame(frame_id: int) -> dict:
    n = random.choices([0,1,2,3,4], weights=[15,35,28,15,7])[0]
    dets = []
    for _ in range(n):
        cls = random.choice(CLASSES)
        x1,y1 = random.randint(0,1000), random.randint(0,600)
        x2,y2 = min(x1+random.randint(40,250),1280), min(y1+random.randint(40,250),720)
        dets.append({
            "track_id": random.randint(1,30),
            "class_name": cls,
            "confidence": round(random.uniform(0.5,0.99),3),
            "bbox_xyxy": [x1,y1,x2,y2],
            "depth_m": round(random.uniform(0.3,10.0),2),
        })
    return {
        "frame_id": frame_id,
        "timestamp_ms": round(time.time()*1000,1),
        "width": 1280, "height": 720,
        "detections": dets,
        "inference_ms": round(random.uniform(6,40),2),
    }

def synthetic_depth(frame: dict) -> dict:
    return {
        "backend": random.choice(["depth_pro","midas"]),
        "depth_min_m": round(random.uniform(0.2,1.0),3),
        "depth_max_m": round(random.uniform(5.0,12.0),3),
        "depth_mean_m": round(random.uniform(1.5,5.0),3),
        "inference_ms": round(random.uniform(15,80),2),
        "per_detection": [
            {"track_id": d["track_id"], "depth_m": d["depth_m"]}
            for d in frame["detections"]
        ],
    }

def synthetic_audio() -> dict | None:
    if random.random() > 0.12: return None
    phrases = [
        "motion detected near entry", "all clear zone two",
        "unknown object approaching", "system nominal",
        "override acknowledged", "intruder alert perimeter",
    ]
    return {"text": random.choice(phrases), "duration_ms": round(random.uniform(600,3500),1)}

# ── Synthetic trainingGUI inference ────────────────────────────────────────────

class SimAdapter:
    """Simulates a trainingGUI adapter — returns structured inference results."""
    def __init__(self, framework: str):
        self.framework = framework
        self._weights = {"temperature": 0.7, "top_p": 0.9, "threshold": 0.5}
        self._call_count = 0

    def run_inference(self, inputs: dict) -> dict:
        self._call_count += 1
        t0 = time.perf_counter()
        # simulate framework-specific latency
        latency_map = {"pytorch":12, "llamacpp":45, "llm_provider":80, "onnx":8, "tensorflow":18}
        base_ms = latency_map.get(self.framework, 20)
        time.sleep(base_ms * 0.001 * random.uniform(0.8, 1.4))

        # simulate output based on input detections
        dets = inputs.get("detections", [])
        if self.framework == "llamacpp":
            output = f"[{self.framework}] {len(dets)} objects observed. Recommend {'caution' if len(dets)>2 else 'monitoring'}."
        elif self.framework == "llm_provider":
            output = {"intent": "monitor" if len(dets)<3 else "escalate", "confidence": round(random.uniform(0.7,0.99),3)}
        else:
            output = {"class_logits": [round(random.uniform(0,1),3) for _ in range(8)],
                      "detection_count": len(dets)}

        return {
            "output": output,
            "latency_ms": round((time.perf_counter()-t0)*1000, 2),
            "framework": self.framework,
            "call_count": self._call_count,
        }

    def get_weight_keys(self) -> list:
        return list(self._weights.keys())

    def set_weight(self, key: str, value) -> None:
        if key in self._weights:
            self._weights[key] = value

    def get_metadata(self) -> dict:
        return {"framework": self.framework, "device": "cpu",
                "model_path": f"~/models/{self.framework}_sim", "adapter_class": "SimAdapter"}

class SimSession:
    """Simulates trainingGUI InferenceSession."""
    def __init__(self, adapter: SimAdapter):
        self.adapter = adapter
        self.history = []
        self._weight_state = dict(adapter._weights)

    def run(self, inputs: dict) -> dict:
        result = self.adapter.run_inference(inputs)
        self.history.append({"ts": time.time(), "inputs": inputs, "result": result})
        return result

    def apply_weight(self, key: str, value) -> None:
        self.adapter.set_weight(key, value)
        self._weight_state[key] = value

    def get_weight_state(self) -> dict:
        return dict(self._weight_state)

# ── Rule-based edge planner ────────────────────────────────────────────────────

def plan(context: dict, inference_result: dict) -> dict:
    dets = context.get("recent_detections", [])
    classes = context.get("classes_seen", {})
    zone = context.get("long_term", {}).get("zone", "public")
    inf_output = inference_result.get("output", {})

    # cross-repo: let inference output influence decision
    escalate = False
    if isinstance(inf_output, dict):
        escalate = inf_output.get("intent") == "escalate"
    elif isinstance(inf_output, str):
        escalate = "caution" in inf_output.lower()

    if "person" in classes and zone == "restricted":
        action, payload, reason = "alert", {"message":"Person in restricted zone","severity":"critical"}, \
            "Person detected in restricted zone — critical alert"
    elif escalate and dets:
        action, payload, reason = "alert", {"message":"Inference model flagged escalation","severity":"high"}, \
            "trainingGUI inference recommends escalation"
    elif len(dets) > 3:
        top = dets[0]
        action, payload, reason = "track", {"track_id":top["track_id"],"label":top["class_name"]}, \
            f"High detection count ({len(dets)}) — tracking primary object"
    elif not dets:
        action, payload, reason = "ignore", {}, "Scene clear — no action needed"
    else:
        action, payload, reason = "log", {"note":f"{list(classes.keys())} at depth {dets[0].get('depth_m','?')}m"}, \
            "Routine observation logged"

    return {"action": action, "payload": payload, "reasoning": reason,
            "decision_ms": round(random.uniform(4,18),1)}

# ── Sim memory ─────────────────────────────────────────────────────────────────

class SimMemory:
    def __init__(self):
        self._dets, self._decisions, self._speech = [], [], []
        self._long = {"zone": random.choice(ZONES), "session": SESSION_ID,
                      "node_id": "dual-sim-01", "alert_cooldown_s": 10}

    def observe(self, det): self._dets.append(det)
    def record(self, d): self._decisions.append(d)
    def hear(self, t): self._speech.append(t)
    def remember(self, k, v): self._long[k] = v

    def context(self) -> dict:
        recent = self._dets[-10:]
        cls = {}
        for d in recent: cls[d["class_name"]] = cls.get(d["class_name"],0)+1
        return {
            "timestamp": time.time(),
            "classes_seen": cls,
            "active_track_ids": list({d["track_id"] for d in recent}),
            "recent_detections": recent[-5:],
            "recent_speech": self._speech[-3:],
            "recent_decisions": [{"action":d["action"],"reasoning":d["reasoning"]} for d in self._decisions[-3:]],
            "long_term": self._long,
        }

    def snapshot(self) -> dict:
        return {"short_term_count": len(self._dets), "decisions_made": len(self._decisions),
                "speech_heard": len(self._speech), "zone": self._long.get("zone"),
                "long_term": self._long}

# ── Cross-system scoring ───────────────────────────────────────────────────────

def cross_score(decision: dict, inference: dict) -> dict:
    action = decision.get("action","ignore")
    output = inference.get("output",{})
    # agreement: both systems flagging same urgency?
    inf_urgent = False
    if isinstance(output, dict): inf_urgent = output.get("intent") == "escalate"
    elif isinstance(output, str): inf_urgent = "caution" in output.lower()
    dec_urgent = action in ("alert",)

    agreement = 1.0 if (inf_urgent == dec_urgent) else 0.0
    latency_delta = round(inference.get("latency_ms",0) - decision.get("decision_ms",0), 2)

    return {
        "agreement_score": agreement,
        "latency_delta_ms": latency_delta,
        "inference_urgent": inf_urgent,
        "decision_urgent": dec_urgent,
        "both_agree": inf_urgent == dec_urgent,
    }

# ── Main loop ──────────────────────────────────────────────────────────────────

def run():
    logger.info("Dual-repo simulation | iters=%d | session=%s | output=%s",
                MAX_ITERS, SESSION_ID, OUTPUT_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # pick a random framework for this session (simulates trainingGUI adapter selection)
    framework = random.choice(FRAMEWORKS)
    logger.info("trainingGUI adapter: %s", framework)
    adapter = SimAdapter(framework)
    session = SimSession(adapter)
    memory = SimMemory()

    last_decision_ts = 0.0
    frame_id = 0
    actuator_log = []

    with open(OUTPUT_PATH, "w") as f:
        for i in range(1, MAX_ITERS+1):
            frame_id += 1
            now = time.time()

            # ── Edge node: perception ──
            frame = synthetic_frame(frame_id)
            depth = synthetic_depth(frame)
            audio = synthetic_audio()
            for det in frame["detections"]:
                memory.observe(det)
            if audio:
                memory.hear(audio["text"])

            # ── trainingGUI: inference ──
            inf_inputs = {
                "detections": frame["detections"],
                "frame_id": frame_id,
                "prompt": f"Analyze {len(frame['detections'])} detections in zone {memory._long['zone']}",
            }
            inf_result = session.run(inf_inputs)

            # ── Agent: decision (every 0.5s sim time) ──
            decision = None
            if now - last_decision_ts >= 0.5:
                last_decision_ts = now
                ctx = memory.context()
                decision = plan(ctx, inf_result)
                memory.record(decision)
                # actuator
                act_record = {"channel": "mqtt" if decision["action"] in ("alert","track","speak") else "internal",
                              "action": decision["action"], "payload": decision["payload"],
                              "timestamp_ms": round(now*1000,1)}
                actuator_log.append(act_record)

                # occasionally adjust trainingGUI weights based on detections
                if random.random() < 0.1:
                    new_temp = round(random.uniform(0.3, 1.2), 2)
                    session.apply_weight("temperature", new_temp)
                    memory.remember("last_weight_update", {"key":"temperature","value":new_temp,"iter":i})

            # ── Cross-system scoring ──
            cross = cross_score(decision or {"action":"ignore"}, inf_result)

            # ── Write JSONL record ──
            record = {
                "iter": i,
                "session_id": SESSION_ID,
                "timestamp_ms": round(now*1000,1),
                "repo": "dual",
                "edge": {
                    "frame_id": frame_id,
                    "inference_ms": frame["inference_ms"],
                    "detection_count": len(frame["detections"]),
                    "detections": frame["detections"],
                    "depth": depth,
                    "audio": audio,
                },
                "inference": {
                    "framework": inf_result["framework"],
                    "output": inf_result["output"],
                    "latency_ms": inf_result["latency_ms"],
                    "call_count": inf_result["call_count"],
                    "weight_state": session.get_weight_state(),
                },
                "decision": decision,
                "actuator_outputs": [actuator_log.pop(0)] if actuator_log else [],
                "cross": cross,
                "memory": memory.snapshot(),
            }
            f.write(json.dumps(record)+"\n")
            f.flush()

            if i % 20 == 0:
                logger.info("iter %d/%d | dets:%d | inf_ms:%.1f | action:%s | agree:%.0f",
                            i, MAX_ITERS, len(frame["detections"]),
                            inf_result["latency_ms"],
                            decision["action"] if decision else "—",
                            cross["agreement_score"]*100)

    logger.info("Done. %d records -> %s", MAX_ITERS, OUTPUT_PATH)

    # print summary stats
    lines = open(OUTPUT_PATH).readlines()
    records = [json.loads(l) for l in lines]
    actions = {}
    agreements = []
    for r in records:
        if r["decision"]:
            a = r["decision"]["action"]
            actions[a] = actions.get(a,0)+1
        agreements.append(r["cross"]["agreement_score"])

    print(f"\n=== Dual Sim Summary ({len(records)} iters, session {SESSION_ID}) ===")
    print(f"Framework: {framework}")
    print(f"Actions: {json.dumps(actions)}")
    print(f"Agreement rate: {sum(agreements)/len(agreements)*100:.1f}%")
    print(f"Avg inference latency: {sum(r['inference']['latency_ms'] for r in records)/len(records):.1f}ms")
    print(f"\nSample record (iter 1):")
    r = records[0]
    print(json.dumps({
        "iter": r["iter"], "framework": r["inference"]["framework"],
        "dets": r["edge"]["detection_count"],
        "inf_output": r["inference"]["output"],
        "decision": r["decision"]["action"] if r["decision"] else None,
        "reasoning": r["decision"]["reasoning"][:70] if r["decision"] else None,
        "agreement": r["cross"]["both_agree"],
    }, indent=2))

if __name__ == "__main__":
    run()
