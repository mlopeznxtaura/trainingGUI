# SDK Report — top500-free-sdks + sdk-clusters-25

NextAura, Inc. — Marco Lopez
Generated from: top500-free-sdks.pdf + sdk-clusters-25.pdf

---

## Document 1: top500-free-sdks.pdf

500 free SDKs across 12 categories. Every entry has a name, vendor, description, and tags (ai/gpu/cloud/web/mobile/data/net/security/game/robot/blockchain).

### Categories and counts

| Category | Count |
|---|---|
| AI & LLM | ~90 |
| GPU & Compute | ~40 |
| Cloud & Infra | ~40 |
| Data & Analytics | ~40 |
| Web & Frontend | ~30 |
| Mobile | 24 |
| Networking | ~25 |
| Security | ~20 |
| Game & XR | ~20 |
| Robotics & IoT | ~20 |
| Blockchain & Web3 | ~15 |
| Other / Cross-category | ~136 |

### Key SDKs by category

**AI & LLM**
- OpenAI Python + Node.js SDKs
- Anthropic Python + TypeScript SDKs
- Google Generative AI SDK (Gemini 1.5 Pro/Flash, free via AI Studio)
- Vertex AI SDK (Gemini, PaLM, Imagen, MLOps)
- Hugging Face Transformers, Hub, PEFT, TRL
- LangChain, LangGraph, LlamaIndex
- Ollama (local LLM REST, Llama3/Mistral/Phi/Gemma)
- Mistral AI SDK
- OpenRouter SDK (100+ models, OpenAI-compatible)
- Together AI SDK
- Semantic Kernel (Microsoft, .NET/Python/Java)
- Haystack (deepset, production RAG)
- DSPy (Stanford, programmatic LLM optimization)
- Instructor (structured outputs via Pydantic)
- CrewAI, AutoGen (multi-agent)
- MLflow (tracking, packaging, deploying)
- NVIDIA Merlin SDK (GPU recommender systems)
- llama-cpp-python / GGUF Tools
- LangSmith SDK (observability)
- Depth Pro SDK (Apple, monocular depth estimation)
- Vercel AI SDK (provider-agnostic streaming UI)

**GPU & Compute**
- CUDA Toolkit, cuBLAS, cuFFT, cuSPARSE, cuRAND
- RAPIDS cuML, cuDF, cuGraph, RAFT
- NVIDIA Warp (differentiable physics/simulation)
- cuQuantum (quantum circuit simulation)
- NVIDIA HPC SDK (C/C++/Fortran HPC)
- OpenCL SDK (cross-platform heterogeneous compute)
- ROCm + HIP (AMD GPU stack)
- Intel SYCL / DPC++, oneMKL
- JAX (NumPy on GPU/TPU, autograd, JIT, vmap, pmap)
- XLA (TF/JAX compiler for GPU/TPU)
- PyTorch (dominant research + production DL)
- CuPy (GPU NumPy/SciPy drop-in)
- Triton (OpenAI, Python-like GPU kernel compilation)
- Numba (Python JIT to GPU/CPU)
- Taichi (GPU programming with autodiff)

**Cloud & Infra**
- AWS Boto3, AWS JS SDK v3
- Google Cloud SDK, Google Cloud Node.js
- Firebase SDK (Web + iOS/Android)
- Pulumi SDK (multi-cloud IaC in real languages)
- Supabase SDK (open-source Firebase alternative)
- Cloudflare Workers SDK (edge compute, D1, KV, R2, AI)
- Vercel SDK, Netlify SDK
- Docker SDK for Python
- Kubernetes Python Client
- Helm SDK
- Temporal SDK Python + TypeScript (durable workflows)
- Neon SDK (serverless Postgres with branching)
- Upstash Redis SDK (serverless Redis over HTTP)
- MinIO SDK (S3-compatible object storage)
- Redis SDK (redis-py)
- Convex SDK (reactive backend-as-a-service)

**Data & Analytics**
- Apache Spark, Kafka, Flink, Beam SDKs
- dbt Core (SQL transformation)
- Airflow, Prefect, Dagster SDKs (orchestration)
- Pandas, Polars, NumPy
- scikit-learn, XGBoost, LightGBM
- DuckDB (in-process OLAP, queries Parquet/CSV/JSON)
- ClickHouse Python Client
- Great Expectations (data quality)
- Prometheus Client
- LanceDB SDK (serverless vector DB, Parquet-native)
- Chroma (open-source embedding DB for RAG)
- Pinecone SDK (managed vector DB)
- Qdrant SDK (high-performance vector search)
- Weaviate SDK (AI-native vector DB)
- Dask (parallel computing, scale Pandas/NumPy)
- Plotly Dash, Panel SDKs (Python dashboards)
- SQLAlchemy (Python SQL toolkit + ORM)
- Apache Arrow (cross-language columnar memory)

**Web & Frontend**
- Next.js, Vue.js, Astro
- Shadcn/ui (React + Radix + Tailwind)
- Apollo Client (GraphQL + React)
- Resend SDK (email API, free 3k/month)
- Sanity Client, Contentful SDK (headless CMS)
- FastAPI, Pydantic, Prisma, Drizzle ORM

**Mobile (24 SDKs)**
- React Native, Flutter, Expo
- iOS SDK (UIKit, SwiftUI, AppKit, CoreML)
- Kotlin Multiplatform, SwiftUI, Jetpack Compose
- Firebase SDK iOS/Android
- Capacitor, NativeScript, Realm
- Amplitude, Mixpanel (analytics)
- OneSignal (push notifications)
- RevenueCat (subscriptions)
- Core ML Tools (convert TF/PyTorch/ONNX to CoreML)

**Networking**
- gRPC SDK, Protocol Buffers
- libp2p (modular P2P networking)
- WebRTC SDK (Google)
- Twilio, Vonage, Agora RTC SDKs
- Pusher, Ably (real-time WebSocket)
- QUIC SDK (quiche, Cloudflare Rust)
- WireGuard SDK, OpenVPN SDK
- Mosquitto MQTT, CoAP (IoT)

**Security**
- Vault SDK (HashiCorp, secrets management)
- Keycloak SDK (OIDC/OAuth2/SAML)
- OpenSSL, libsodium, Tink (Google)
- Falco (runtime container security)
- Semgrep, Snyk, Nuclei, OWASP ZAP
- Auth0 SDK, Passkeys/WebAuthn, JWT

---

## Document 2: sdk-clusters-25.pdf

25 clusters, each with 20 SDKs, a buildable product concept, and a concrete starting point. Total: 500 SDK-slot assignments across 25 product ideas.

### All 25 clusters

| # | Name | One-line concept |
|---|---|---|
| 01 | Sentient Edge Node | Local AI agents — perceive, decide, act, no cloud |
| 02 | Physics-Native AI Trainer | Train robot policies in GPU-accelerated simulation |
| 03 | Sovereign Developer OS | Fully local self-improving AI dev environment |
| 04 | Spatial Operating System | OpenUSD-native app platform — UI is a room, not a window |
| 05 | Bio-Compute Stack | GPU drug discovery + genomics pipeline, fully OSS |
| 06 | Autonomous Video Intelligence | Real-time multi-camera scene understanding at the edge |
| 07 | Neural Audio Forge | Voice cloning, music gen, spatial sound pipeline |
| 08 | Distributed Simulation Fabric | Planet-scale event-driven simulation, thousands of agents |
| 09 | Quantum-Classical ML Bridge | Hybrid quantum-classical models for optimization |
| 10 | Generative World Engine | Procedural world generation for synthetic training data |
| 11 | Embedded Intelligence Mesh | TinyML fleet on microcontrollers with OTA learning |
| 12 | Sovereign Data Warehouse | Self-hosted GPU-accelerated Snowflake/Databricks replacement |
| 13 | Real-Time Collaborative Mesh | Multiplayer-native app infra — no backend required |
| 14 | GPU Kernel Workshop | Write, profile, and optimize custom GPU kernels from Python |
| 15 | Zero-Trust Security Fabric | Self-hosted identity, secrets, runtime security |
| 16 | Digital Human Platform | Real-time generative AI NPCs with voice, face, memory |
| 17 | Autonomous Manufacturing Agent | AI-native production floor — perception, planning, quality |
| 18 | Hyper-Personalization Engine | Real-time GPU recommendation system |
| 19 | Geospatial Intelligence Platform | Planetary-scale spatial analysis and visualization |
| 20 | Open Science Compute Stack | Physics sim + FEM + CFD on GPU, research-grade, free |
| 21 | Decentralized AI Inference Network | P2P trustless model serving network |
| 22 | Web3 Developer Toolkit | Full-stack dApp — contract to UI |
| 23 | Mobile AI Platform | On-device AI for iOS/Android — no cloud inference |
| 24 | Full-Stack AI SaaS Boilerplate | Ship an AI product in a week |
| 25 | Autonomous Research Agent | Self-directed AI that reads, hypothesizes, experiments, writes |

### Cluster deep-dives

**01 — Sentient Edge Node**
SDKs: Ollama, llama.cpp, LangGraph, NVIDIA Warp, OpenCV, YOLO (Ultralytics), FastAPI, SQLAlchemy, Pydantic AI, DALI, MediaPipe, Mosquitto MQTT, TensorFlow Lite, Open3D, Prometheus Client, PortAudio, Depth Pro, ZeroMQ, SAM2, OpenTelemetry
Start: perception/vision.py — YOLO on camera frames, structured detection JSON output. Nothing else first.

**02 — Physics-Native AI Trainer**
SDKs: MuJoCo, NVIDIA Isaac Lab, NVIDIA Warp, Gymnasium, Stable-Baselines3, PyTorch, cuDNN, W&B, Pinocchio, cuRobo, Drake, HF TRL, RAPIDS RAFT, Ray, MLflow, Numba, SymPy, Open3D, Gaussian Splatting, nerfstudio
Start: MuJoCo env wrapped as Gymnasium env, SB3 PPO training, logged to W&B. Simple closed loop first.

**03 — Sovereign Developer OS**
SDKs: Ollama, LangGraph, LlamaIndex, Chroma, Unsloth, Axolotl, ExLlamaV2, Tauri, SQLAlchemy, FastAPI, Pydantic, Instructor, DSPy, GGUF Tools, W&B, GitHub Actions SDK, Prometheus Client, OpenTelemetry, Nx Build System, Nix SDK
Start: core/rag.py — LlamaIndex pipeline indexing local code to Chroma, query via Ollama.

**04 — Spatial Operating System**
SDKs: OpenUSD, NVIDIA Omniverse, Apple visionOS, ARKit, OpenXR, WebXR, React Three Fiber, Three.js, NVIDIA Warp, NVIDIA PhysX, Babylon.js XR, LiveKit, ONNX Runtime, MediaPipe, Framer Motion, Zustand, FastAPI, Supabase, Socket.io, A-Frame
Start: research loading .usda in Three.js before writing any code.

**05 — Bio-Compute Stack**
SDKs: BioNeMo, NVIDIA Clara Parabricks, MONAI, RDKit, BioPython, OpenMM, GROMACS, Scanpy, PyTorch, HF Transformers, JAX, NumPy, Polars, DuckDB, Apache Arrow, W&B, FastAPI, Prefect, PennyLane, SymPy
Start: molecular/protein_fold.py — amino acid sequence to ESMFold inference, save as .pdb.

**06 — Autonomous Video Intelligence**
SDKs: NVIDIA DeepStream, NVIDIA Video Codec SDK, YOLO, SAM2, OpenCV, GStreamer, TensorRT, NVIDIA Triton, Kafka, Redis, FastAPI, Prometheus Client, Grafana, TimescaleDB, LiveKit, Socket.io, Cloudflare Workers, Supabase, OpenTelemetry, NATS
Start: inference/detector.py — single camera, YOLO, structured JSON with track IDs at max FPS.

**07 — Neural Audio Forge**
SDKs: Whisper, Coqui TTS, ElevenLabs SDK, Deepgram, librosa, Essentia, PortAudio, NVIDIA Riva, NVIDIA Audio2Face, AssemblyAI, Tone.js, Web Audio API, FFmpeg, GStreamer, LiveKit, FastAPI, Redis, Supabase, Cloudinary, Socket.io
Start: transcription/whisper_local.py — video/audio to Whisper, segments with precise timing.

**08 — Distributed Simulation Fabric**
SDKs: Ray, NVIDIA Warp, MuJoCo, Gymnasium, LangGraph, Kafka, Apache Flink, Redis, NATS, gRPC, Temporal (Python), DuckDB, Apache Arrow, Polars, Prometheus Client, Grafana, FastAPI, Pydantic, CuPy, Numba
Start: Gymnasium env with 100 agents stepping in parallel via Ray, logging to stdout. No Kafka yet.

**09 — Quantum-Classical ML Bridge**
SDKs: PennyLane, Qiskit, Cirq, cuQuantum, PyTorch, JAX, Flax, NumPy, SciPy, SymPy, W&B, MLflow, Ray, FastAPI, Polars, DuckDB, HF Hub, Pydantic, Prometheus Client, DSPy
Start: 4-qubit PennyLane variational circuit as a PyTorch nn.Module, trained end-to-end on a toy classifier.

**10 — Generative World Engine**
SDKs: NVIDIA Cosmos, NVIDIA Omniverse, OpenUSD, Gaussian Splatting, nerfstudio, NVIDIA Warp, NVIDIA PhysX, Omniverse Replicator, HF Diffusers, ControlNet, DALI, Blender Python API, OpenCV, Depth Pro, PyTorch, FastAPI, MinIO, Prefect, W&B, Apache Arrow
Start: Omniverse Replicator — 10 scene variations, PNG + JSON annotation output.

**11 — Embedded Intelligence Mesh**
SDKs: Edge Impulse, TF Lite Micro, Arduino SDK, ESP-IDF, Zephyr RTOS, MicroPython, CircuitPython, Mosquitto MQTT, CoAP, Nordic nRF SDK, Arm CMSIS, FreeRTOS, Balena, AWS IoT, Prometheus Client, TimescaleDB, Grafana, FastAPI, Redis, Nix SDK
Start: Edge Impulse ESP32 vibration anomaly detector — full cycle from data collection to on-device inference.

**12 — Sovereign Data Warehouse**
SDKs: DuckDB, Apache Arrow, Polars, Apache Spark, Kafka, Apache Flink, dbt Core, RAPIDS cuDF, RAPIDS cuML, Dagster, Prefect, MinIO, Great Expectations, OpenTelemetry, Prometheus Client, Grafana, FastAPI, Plotly Dash, Streamlit, SQLAlchemy
Start: MinIO Parquet writes + DuckDB querying Parquet on MinIO directly. Connect these two first.

**13 — Real-Time Collaborative Mesh**
SDKs: LiveKit, Socket.io, NATS, Ably, Pusher, Supabase, Convex, Upstash Redis, Cloudflare Workers, Vercel, Next.js, React, Zustand, TanStack Query, Framer Motion, tRPC, Auth.js, Zod, WebRTC, gRPC
Start: Next.js multi-tab synchronized counter via Supabase Realtime.

**14 — GPU Kernel Workshop**
SDKs: CUDA Toolkit, Triton (OpenAI), CuPy, Numba, NVIDIA Warp, Taichi, JAX, cuBLAS, cuFFT, cuSPARSE, SPIRV-Cross, Slang SDK, OpenCL, ROCm, Nsight Compute, RenderDoc, W&B, Prometheus Client, FastAPI, Plotly Dash
Start: Triton blocked matrix multiplication, benchmark vs torch.matmul (cuBLAS), understand tuning knobs first.

**15 — Zero-Trust Security Fabric**
SDKs: Vault, Keycloak, OpenSSL, libsodium, Python Cryptography, Tink, Falco, Semgrep, Nuclei, OWASP ZAP, Snyk, Auth0, Passkeys/WebAuthn, JWT, WireGuard, Kubernetes Python Client, Docker Python SDK, Prometheus Client, OpenTelemetry, Grafana
Start: secrets/vault_client.py — AppRole auth, get/set secret, generate DB creds, token renewal.

**16 — Digital Human Platform**
SDKs: NVIDIA ACE, Audio2Face, NVIDIA Riva, Coqui TTS, Whisper, LangGraph, LlamaIndex, Chroma, Ollama, SAM2, MediaPipe, LiveKit, WebRTC, React Three Fiber, Three.js, Socket.io, Supabase, FastAPI, Redis, Pydantic AI
Start: npc/brain.py — LangGraph character with session memory, in-character response via Ollama.

**17 — Autonomous Manufacturing Agent**
SDKs: NVIDIA DeepStream, YOLO, OpenCV, NVIDIA VPI, NVIDIA Isaac Perceptor, cuRobo, MoveIt, ROS 2, LangGraph, Ollama, Kafka, TimescaleDB, FastAPI, Grafana, Prometheus Client, Redis, NATS, Temporal (Python), W&B, Great Expectations
Start: perception/inspection.py — camera to YOLO to pass/fail to structured inspection result. Clean output before Kafka.

**18 — Hyper-Personalization Engine**
SDKs: NVIDIA Merlin, RAPIDS cuML, RAPIDS cuDF, HugeCTR, HF Transformers, Pinecone, Qdrant, LanceDB, Kafka, Apache Flink, Redis, FastAPI, Polars, DuckDB, W&B, MLflow, Prometheus Client, Grafana, Prefect, OpenTelemetry
Start: NVTabular — user-item interaction CSV, categorical encoding, output ready for HugeCTR.

**19 — Geospatial Intelligence Platform**
SDKs: Deck.gl, Kepler.gl, H3, PostGIS, GDAL, GeoPandas, Cesium, Mapbox GL JS, Maplibre GL JS, Turf.js, PyProj, OpenStreetMap SDK, DuckDB, Apache Arrow, Polars, Kafka, FastAPI, Redis, Prometheus Client, HF Transformers
Start: spatial/h3_index.py — (lat, lon, value) tuples to H3 cells, aggregate, return GeoDataFrame.

**20 — Open Science Compute Stack**
SDKs: OpenFOAM, FEniCS, OpenMM, GROMACS, OpenSim, VTK, ITK, NVIDIA Warp, CuPy, JAX, NumPy, SciPy, SymPy, AstroPy, Polars, DuckDB, Apache Arrow, W&B, FastAPI, Plotly Dash
Start: fem/fenics_solver.py — Poisson equation on unit square, output as numpy + VTK.

**21 — Decentralized AI Inference Network**
SDKs: libp2p, IPFS, Ollama, vLLM, LiteLLM, TensorRT, ONNX Runtime, gRPC, NATS, libsodium, Vault, WireGuard, Prometheus Client, FastAPI, Redis, Pydantic, ethers.js, The Graph SDK, Protocol Buffers, OpenTelemetry
Start: two local libp2p nodes, discovery, route Ollama inference via gRPC. Local network only first.

**22 — Web3 Developer Toolkit**
SDKs: Foundry, Hardhat, OpenZeppelin, viem, wagmi, RainbowKit, ethers.js, Alchemy SDK, The Graph, IPFS, zkSync, Polygon, Thirdweb, WalletConnect, Next.js, React, TanStack Query, Zod, tRPC, Supabase
Start: Foundry project, minimal ERC20 with OpenZeppelin, test mint + transfer on local Anvil.

**23 — Mobile AI Platform**
SDKs: Core ML Tools, Metal Performance Shaders, Apple MLX, ARKit, SwiftUI, TF Lite, Android SDK, Jetpack Compose, ARCore, MediaPipe, ONNX Runtime, React Native, Expo, Vision Camera, Reanimated, Firebase iOS/Android, RevenueCat, ElevenLabs, Deepgram, Whisper
Start: Vision Camera frame processor — MobileNet classifier on each frame, top-1 label + confidence.

**24 — Full-Stack AI SaaS Boilerplate**
SDKs: Next.js, React, Tailwind CSS, Shadcn/ui, Clerk, Stripe, Resend, Supabase, Prisma, tRPC, Zod, Anthropic TypeScript SDK, OpenAI Node.js SDK, LangChain JS, Upstash Redis, Cloudflare Workers, Vercel, PostHog, Sentry, LangSmith
Start: streaming AI endpoint with Upstash rate limiting, Anthropic streaming, Vercel AI SDK useChat compatibility.

**25 — Autonomous Research Agent**
SDKs: LangGraph, LlamaIndex, DSPy, AutoGen, CrewAI, Pydantic AI, Instructor, Anthropic Python SDK, OpenAI Python SDK, Ollama, Chroma, Qdrant, W&B, MLflow, Prefect, FastAPI, Supabase, Redis, LangSmith, Gradio
Start: knowledge/ingestion.py — PDF ingest with section awareness, embed with Ollama nomic-embed-text, store in Chroma with metadata.

---

## Cross-Reference: trainingGUI overlap

SDKs already implemented in trainingGUI:
- PyTorch (cluster 02, 05, 09, 10, 14, many others)
- JAX (cluster 02, 05, 09, 20)
- TensorFlow / TF Lite (cluster 11, 23)
- ONNX Runtime (cluster 04, 21, 23)
- llama.cpp / llama-cpp-python (cluster 01, 21)
- Ollama (cluster 01, 03, 16, 17, 21, 25)
- MLflow (cluster 02, 09, 25)
- NiceGUI (not in clusters — trainingGUI-specific)

Highest-value SDKs from the list NOT yet in trainingGUI:
- vLLM (cluster 21 — production LLM serving, high-throughput)
- LiteLLM (cluster 21 — unified API across all providers)
- TensorRT (cluster 06, 21 — NVIDIA optimized inference)
- NVIDIA Triton (cluster 06 — multi-framework model serving)
- Hugging Face TRL (cluster 02 — RLHF/DPO/SFT training loops)
- W&B SDK (cluster 02, 09, 10, 14, 25 — experiment tracking)
- Ray SDK (cluster 02, 08, 09 — distributed execution)
- LangGraph (cluster 01, 03, 08, 16, 17, 25 — stateful agent graphs)
- LlamaIndex (cluster 03, 16, 25 — RAG + data framework)
- DSPy (cluster 03, 09, 25 — programmatic LLM optimization)

---

*Report generated by folk AI from top500-free-sdks.pdf and sdk-clusters-25.pdf*
*Author: NextAura, Inc. — Marco Lopez*
