# MASTER ENGINEERING PROMPT — REBUILD MULTICHEAT INTO A PROFESSIONAL, HIGH-ACCURACY CLASSROOM MALPRACTICE DETECTION SYSTEM

You are working on an existing project called **MulTiCheat**, an AI-based classroom/exam cheating detection system.

Your task is **not to make cosmetic improvements** and not merely to tune existing thresholds.

Your task is to **deeply audit the complete existing repository and then redesign, refactor, implement, test and validate the computer-vision pipeline so that the system becomes a production-grade, evidence-based classroom malpractice detection system capable of monitoring approximately 50–60 students from CCTV footage in a single classroom**.

The existing codebase, architecture, APIs and frontend should be preserved where useful, but weak components must be replaced rather than protected for compatibility.

The final result must be a genuinely functioning system, not a demo with hardcoded detections or heuristics that merely appear convincing.

---

# 1. EXISTING PROJECT — USE THIS AS THE STARTING POINT

The current system contains:

- FastAPI backend
- React + TypeScript + Vite frontend
- OpenCV video ingestion and streaming
- YOLOv8 object detection
- YOLOv8-pose
- BoT-SORT tracking
- motion detection
- geometric interaction analysis
- head-yaw heuristics
- wrist-distance heuristics
- paper-copying raycasting
- temporal sliding-window aggregation
- evidence-frame generation
- JSON/CSV/ZIP exports
- canvas-based frontend overlays

The current system is described as complete at the application level, but its actual cheating detection needs substantial improvement. In particular, the current detector:

- samples video approximately every 0.5 seconds
- uses generic COCO classes such as `person`, `cell phone`, and `book`
- uses centroid containment to associate objects with students
- estimates looking behavior using nose-vs-shoulder geometry
- estimates suspicious hand movement using wrist/shoulder ratios
- estimates copying using lateral head-yaw raycasting
- uses fixed spatial thresholds
- aggregates detections after the fact

These mechanisms are inadequate for high-accuracy detection involving 50–60 students.

Do not blindly retain them.

Treat the current codebase as an existing prototype that must be upgraded into a proper perception system.

---

# 2. PRIMARY OBJECTIVE

Build a system that can process a typical classroom/exam CCTV recording containing approximately **50–60 students**, including students located:

- in the foreground
- middle rows
- rear rows
- corners
- partially occluded positions
- small image regions
- different lighting conditions
- different seating orientations

The system must maintain an **anonymous persistent student track identity** for every visible student.

Do not rely on face recognition for student identification.

A student should instead be represented by a persistent anonymous identifier such as:

`StudentTrack_023`

and optionally associated with:

`Seat_A12`

or another manually/configured seating location.

The system must distinguish between:

1. Student detection
2. Student tracking
3. Student state estimation
4. Object detection
5. Interaction detection
6. Temporal behavior recognition
7. Malpractice classification
8. Evidence generation
9. Confidence estimation
10. Proctor review

Do not collapse all of these into a single heuristic function.

---

# 3. IMPORTANT ACCURACY REQUIREMENT

Do NOT claim that AI can literally guarantee 100% detection of every malpractice event.

Instead, engineer the system so that:

- every visible student is continuously monitored
- missed detections are measured
- false positives are measured
- track identity switches are measured
- every malpractice category has independent validation metrics
- uncertain events are explicitly classified as uncertain
- the system never converts weak evidence into a definitive accusation

The system's objective is:

**high recall + high precision + strong evidence + calibrated confidence + human review**

rather than generating large numbers of false accusations.

---

# 4. FIRST TASK — COMPLETE REPOSITORY AUDIT

Before changing the architecture:

Inspect the entire repository.

Read every:

- Python file
- TypeScript file
- CSS file
- YAML configuration
- requirements file
- README
- test
- model-loading component
- analysis component
- evidence component
- export component

Create an internal understanding of:

- data flow
- dependency graph
- model lifecycle
- frame processing flow
- tracking lifecycle
- annotation lifecycle
- API lifecycle
- frontend lifecycle
- analysis result schema
- event schema
- configuration schema

Identify:

- dead code
- duplicate functionality
- fragile assumptions
- hardcoded thresholds
- incorrect model assumptions
- inconsistent coordinates
- frame-rate assumptions
- ID-switch vulnerabilities
- object attribution errors
- memory leaks
- synchronous bottlenecks
- GPU/CPU bottlenecks
- error handling deficiencies
- race conditions
- incorrect confidence handling
- poor video decoding behavior
- poor image scaling
- frontend/backend mismatches

Do not start implementation until you understand the existing architecture.

---

# 5. MODERNIZE THE PERCEPTION STACK

Replace the outdated YOLOv8-first design with a modern model strategy.

Evaluate the current production-capable model stack and implement the best practical combination.

Use modern Ultralytics models where appropriate, with **YOLO26** as the primary candidate for detection/pose/segmentation because it is currently the newest Ultralytics family and supports detection, segmentation, depth, classification, pose and OBB.

Do not assume the largest model is automatically best.

Implement a configurable model profile system:

`FAST`

`BALANCED`

`ACCURACY`

`BENCHMARK`

Example:

```yaml
models:
  detector:
    model: yolo26m.pt
    imgsz: 1280
    confidence: 0.20

  pose:
    model: yolo26m-pose.pt

  segmentation:
    enabled: true

  tracker:
    type: botsort
    reid: true

  refinement:
    enabled: true
```

Allow models to be changed entirely through configuration.

Never hardcode model filenames in source code.

---

# 6. SOLVE THE REAR-ROW PROBLEM PROPERLY

This is one of the most important requirements.

The current system attempts to solve small-object detection primarily by running detection at a larger image size. That is insufficient.

A wide classroom CCTV frame containing 50–60 students creates a severe scale problem:

A student occupying 20×40 pixels cannot be detected with the same reliability as a student occupying 200×400 pixels.

Implement **perspective-aware multi-scale inference**.

The pipeline must support:

## A. Full-frame pass

Run detection across the full CCTV frame.

Purpose:

- global student discovery
- classroom geometry
- coarse object detection
- initial tracking

## B. Spatial tiling

Divide the image into overlapping tiles.

For example:

```text
1920×1080

FULL FRAME
      +
4–12 overlapping tiles
      +
adaptive high-resolution crops
```

The tile count must be dynamically configurable.

Use overlap sufficient to prevent students/objects at tile boundaries from disappearing.

Perform detection on each tile.

Map tile coordinates back into original-frame coordinates.

Apply cross-tile duplicate suppression/association.

## C. Student-centric crop inference

After detecting a student, generate a high-resolution crop around that student's body/head/desk region.

Run specialized detectors against the crop.

This is essential.

The system should not ask one detector to simultaneously solve:

- person detection
- tiny phone detection
- paper detection
- chit detection
- hand-object interactions

inside an enormous wide-angle classroom frame.

Instead:

```text
CLASSROOM FRAME
      ↓
PERSON DETECTION
      ↓
TRACK STUDENTS
      ↓
STUDENT CROP
      ↓
HIGH-RES OBJECT + HAND + PAPER DETECTION
      ↓
TEMPORAL ANALYSIS
```

This is the fundamental architecture change.

---

# 7. CREATE A CLASSROOM CALIBRATION STAGE

Before analysis starts, determine classroom geometry.

The user should be able to configure:

- classroom boundary
- board area
- floor area
- desks
- student seating regions
- desk regions
- camera position
- approximate perspective
- rows
- seat coordinates

Create a calibration UI.

Allow the proctor to draw:

- seat regions
- desk regions
- restricted areas
- aisle regions

Automatically estimate perspective and student scale across rows.

For example:

```text
Row 1 → expected student size = large
Row 2 → medium-large
Row 3 → medium
Row 4 → small
Row 5 → very small
```

Use this information to dynamically alter:

- crop size
- detector resolution
- detection thresholds
- NMS/association thresholds
- expected object size
- pose confidence thresholds

Do not use identical thresholds for all students regardless of image scale.

---

# 8. STUDENT-FIRST TRACKING ARCHITECTURE

The primary tracked entity must be the **student**, not the phone or book.

Create a robust `StudentTrack` abstraction.

Example:

```python
StudentTrack(
    track_id,
    bbox,
    centroid,
    confidence,
    pose,
    appearance_embedding,
    seat_id,
    visibility,
    occlusion_state,
    temporal_features,
    risk_state
)
```

Track identity must remain stable through:

- temporary occlusion
- students crossing limbs
- partial occlusion
- short detection failures
- head movement
- body movement
- brief obstruction

Use a modern tracker with:

- motion estimation
- IoU matching
- appearance/ReID features
- track confirmation
- lost-track buffering
- re-identification
- duplicate suppression

BoT-SORT supports optional ReID and camera-motion compensation, while the current Ultralytics tracking stack also provides alternatives including Deep OC-SORT and TrackTrack. Evaluate them empirically rather than assuming one tracker is universally best.

Track performance must be benchmarked using:

- IDF1
- HOTA
- MOTA
- ID switches
- track fragmentation
- detection recall

Do not declare tracking “working” simply because boxes display on the screen.

---

# 9. DO NOT USE A SINGLE PERSON BOUNDING BOX FOR EVERYTHING

Each student needs multiple semantic regions.

Maintain:

```text
HEAD
FACE / HEAD ORIENTATION REGION
UPPER BODY
LEFT ARM
RIGHT ARM
LEFT HAND
RIGHT HAND
DESK
PAPER
LAP / LOWER DESK
PHONE AREA
NEIGHBOR REGION
```

Pose should supply body keypoints.

Then create derived regions dynamically.

For example:

```text
head_roi
left_hand_roi
right_hand_roi
desk_roi
paper_roi
neighbor_left_roi
neighbor_right_roi
```

These regions must move with the tracked student.

---

# 10. CUSTOM OBJECT DETECTION IS MANDATORY

Do NOT rely on generic COCO `cell phone` and `book` labels for final malpractice detection.

The current system is fundamentally limited because generic classes do not represent the exact objects required for exam supervision.

Create a custom exam-malpractice object taxonomy.

At minimum:

```text
phone
smartphone_screen
earbuds
earbud_case
cheat_chit
small_note
printed_note
folded_paper
loose_paper
textbook
reference_book
calculator
writing_paper
answer_sheet
question_paper
pen
pencil
eraser
hand
desk
```

If objects cannot be reliably detected from the wide camera, detect them inside high-resolution student crops.

---

# 11. BUILD A REAL DATASET PIPELINE

Do not expect an off-the-shelf model to accurately recognize custom cheating artifacts.

Create:

```text
dataset/
    raw/
    frames/
    crops/
    annotations/
    train/
    val/
    test/
    hard_negatives/
```

Create annotation tooling.

Annotation types should include:

- student boxes
- phone boxes
- cheat-sheet boxes
- paper boxes
- hands
- desks
- relevant objects
- behavior labels
- interaction labels

Include difficult examples:

- phone partially hidden
- phone in lap
- tiny phone
- phone under paper
- phone in hand
- folded chit
- white chit on white paper
- printed notes
- hand reaching to another desk
- students naturally turning around
- students talking but not cheating
- students looking sideways naturally
- students picking up a pen
- students scratching their head
- students adjusting clothing
- students stretching
- students asking proctor something

Hard negatives are essential.

---

# 12. MALPRACTICE CLASSIFICATION MUST BECOME TEMPORAL

This is the second major architectural change.

Do NOT classify cheating from one frame.

For every student, maintain a temporal state window.

For example:

```text
Track_023

t-5s
t-4.5s
t-4s
...
t
```

Store:

- pose sequence
- head direction
- body orientation
- hand trajectories
- object trajectories
- desk-relative coordinates
- neighbor-relative coordinates
- visibility
- motion
- object confidence
- interaction confidence
- posture
- screen visibility
- paper visibility

Then classify the entire sequence.

Use a temporal model where practical:

- temporal transformer
- temporal convolution
- ST-GCN on pose sequences
- lightweight action-recognition model
- learned sequence classifier

The system must learn that:

```text
looking left once
≠ cheating
```

while:

```text
repeated leftward gaze
+
head rotation
+
neighbor paper region
+
body orientation
+
sustained duration
+
consistent spatial relationship
```

can become:

`LIKELY_PAPER_COPYING`

This should be a learned/aggregated temporal decision, not one threshold.

---

# 13. IMPLEMENT EXPLICIT MALPRACTICE CLASSES

At minimum, support:

### CLASS 1 — PHONE USAGE

Detect:

- phone visible
- phone held
- phone in lap
- phone near hand
- student looking at phone
- repeated phone interaction
- phone screen orientation
- hand-to-phone interaction

Do not flag merely because a phone-shaped object is present.

Classification should require contextual evidence.

Possible confidence composition:

```text
phone detection
+
hand-phone proximity
+
head orientation
+
phone persistence
+
interaction duration
```

Example final event:

```json
{
  "type": "phone_usage",
  "track_id": "StudentTrack_023",
  "confidence": 0.94,
  "start": 183.4,
  "end": 190.8
}
```

---

# 14. CLASS 2 — PEeking / PAPER COPYING

This category must be completely redesigned.

Do not use:

```text
head yaw → nearest student → copying
```

because that produces false positives.

Instead model:

```text
Student A head direction
Student A gaze direction estimate
Student A body orientation
Student A desk location
Student B desk/paper region
angle from A toward B's paper
distance to B
duration
number of repeated glances
student's own paper visibility
```

Define a `PaperCopyInteraction`.

Example:

```python
PaperCopyInteraction(
    source_student=A,
    target_student=B,
    source_head_direction,
    target_paper_region,
    angular_alignment,
    distance,
    duration,
    repetition_count,
    confidence
)
```

Require temporal persistence.

One sideways glance:

`NOT CHEATING`

Repeated directional glances toward a neighboring student's answer region:

`SUSPICIOUS`

Repeated sustained behavior with strong spatial and temporal evidence:

`LIKELY_PAPER_COPYING`

---

# 15. CLASS 3 — CHEAT CHIT / NOTES

Detect:

- presence of unauthorized note
- note held in hand
- note opened/read
- note below desk
- note passed
- note exchanged
- student repeatedly looking at note

Distinguish:

```text
official answer sheet
question paper
rough sheet
```

from:

```text
unauthorized note
cheat chit
reference material
```

Use contextual rules and, where practical, OCR/visual content analysis.

Do not classify every piece of paper as cheating.

---

# 16. CLASS 4 — NOTE PASSING / OBJECT PASSING

Replace the current simple centroid-distance rule.

The current design creates `note_passing` based mainly on the spatial distance between student centroids.

This is not sufficient.

Instead detect:

```text
hand A
        ↓
object
        ↓
hand B
```

across a temporal sequence.

Track the object.

Expected pattern:

```text
Object near A
→ object moves from A
→ object enters shared interaction region
→ object approaches B hand
→ object near B
→ object leaves A
```

This is evidence of transfer.

A simple student-centroid distance must never independently classify note passing.

---

# 17. CLASS 5 — TURNING AROUND

Create a temporal body-orientation classifier.

Detect:

- head rotation
- shoulder rotation
- torso rotation
- direction change
- duration
- recurrence

Distinguish:

```text
brief look backward
```

from:

```text
sustained full-body turn
```

and from:

```text
student naturally adjusting posture
```

Use configurable duration thresholds calibrated to video FPS.

---

# 18. CLASS 6 — LOOKING AROUND

Separate:

`look_around`

from:

`paper_copying`

from:

`talking`

from:

`normal movement`

Use:

- head orientation
- pose
- temporal pattern
- target direction
- neighbor proximity
- target paper visibility

A student should NOT be flagged for malpractice simply because they look left or right.

---

# 19. CLASS 7 — TALKING / COMMUNICATION

Where visual evidence supports it, detect:

- face-to-face orientation
- repeated mutual head orientation
- proximity
- mouth/face motion where resolution permits
- posture toward another student
- repeated interaction pattern

Do not confidently claim speech purely from visual data.

Use:

`possible_communication`

rather than definitive speech classification when evidence is weak.

Audio analysis can be implemented as an optional second channel later.

---

# 20. HAND-BEHAVIOR ANALYSIS

The current `suspicious_hand_movement` heuristic is too broad because a wrist extending beyond a fixed shoulder-relative distance does not mean cheating.

Replace it with:

```text
left hand trajectory
right hand trajectory
hand-to-paper interaction
hand-to-phone interaction
hand-to-note interaction
hand-to-neighbor interaction
hand disappearance
hand reappearance
```

Classify movement based on context.

Examples:

```text
hand moves to own answer sheet
→ normal

hand repeatedly reaches under desk
→ suspicious

hand approaches neighboring student
+
object transfer
→ likely note passing

hand approaches phone
+
head points toward phone
→ likely phone usage
```

---

# 21. USE SEGMENTATION FOR AMBIGUOUS OBJECTS

For difficult objects, incorporate modern video segmentation/refinement.

SAM 3 supports text/exemplar/visual prompting to detect, segment and track matching concepts through video, while SAM 3.1 adds efficient multi-object video tracking.

Do not run expensive segmentation indiscriminately across every pixel of every frame.

Use it as a **second-stage refinement engine**:

```text
cheap detector
      ↓
candidate object
      ↓
SAM refinement
      ↓
precise object mask
      ↓
interaction analysis
```

Use segmentation for:

- tiny paper/chit
- partially occluded phone
- object held by hand
- ambiguous object boundaries
- difficult transfer events

---

# 22. FRAME PROCESSING MUST BE REDESIGNED

The current system's approximately 0.5-second frame sampling creates a fundamental temporal-resolution limitation.

Do not process every frame with every expensive model.

Instead implement an intelligent multi-rate pipeline.

Example:

```text
30 FPS CCTV

TRACKING:
every frame or high-frequency lightweight inference

PERSON DETECTION:
5–15 FPS depending on quality

POSE:
5–15 FPS

OBJECT DETECTION:
adaptive 5–15 FPS

TEMPORAL MODEL:
continuous rolling window

EXPENSIVE REFINEMENT:
only candidate events
```

The exact rates must be configurable.

Never hardcode 2 FPS as the permanent architecture.

---

# 23. EVENT CANDIDATE SYSTEM

Every student should have a lightweight state machine:

```text
NORMAL
↓
OBSERVED
↓
SUSPICIOUS
↓
CANDIDATE
↓
CONFIRMED
↓
RESOLVED
```

Example:

```text
phone confidence = 0.81
      +
hand-phone interaction = 0.72
      +
gaze-to-phone = 0.85
      +
duration = 4.2s

→ candidate

temporal consistency remains high

→ confirmed
```

This avoids instantaneous false flags.

---

# 24. CONFIDENCE MUST BE CALIBRATED

Do not display raw detector confidence as final cheating confidence.

Separate:

```text
detection_confidence
tracking_confidence
behavior_confidence
interaction_confidence
event_confidence
```

Then create a calibrated final probability.

Implement calibration using validation data.

Possible methods:

- temperature scaling
- isotonic regression
- Platt scaling

The UI should display:

`94% event confidence`

only if that value has actually been calibrated.

---

# 25. INTRODUCE EVIDENCE-BASED FLAGGING

Every malpractice event must contain evidence.

Example:

```json
{
  "event_id": "evt_00821",
  "student_id": "StudentTrack_023",
  "type": "paper_copying",
  "confidence": 0.91,
  "start_time": 183.4,
  "end_time": 189.9,
  "evidence": {
      "head_direction": "left",
      "target_student": "StudentTrack_024",
      "target_paper_overlap": 0.78,
      "duration": 6.5,
      "repeated_glances": 7
  }
}
```

Store:

- start frame
- peak frame
- end frame
- 3–10 second evidence clip where appropriate
- annotated frame
- student crop
- neighboring-student crop
- relevant object crop
- confidence timeline

---

# 26. GENERATE EVENT CLIPS

Do not depend exclusively on JPEG evidence frames.

For confirmed/candidate events generate short evidence clips:

```text
5 sec before event
+
event duration
+
5 sec after event
```

Where video format and storage allow it.

Create:

```text
evidence/
    event_0001/
        before.mp4
        event.mp4
        after.mp4
        peak.jpg
        annotated.jpg
        student_crop.jpg
        metadata.json
```

The proctor should be able to review the event instantly.

---

# 27. CREATE A PER-STUDENT TIMELINE

The frontend should not only show global events.

Create:

```text
STUDENT TRACK 001
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
00:00  normal
00:46  looked left
01:20  phone candidate
03:12  normal
05:41  possible paper copying
```

Every student should have an individual behavioral timeline.

The proctor must be able to select a student and review all events associated with that track.

---

# 28. CREATE A CLASSROOM OVERVIEW MODE

Display all currently tracked students simultaneously.

Example:

```text
Student 01  ● Normal
Student 02  ● Normal
Student 03  ● Suspicious
Student 04  ● Normal
...
Student 58  ● Candidate
```

Also provide a classroom view showing:

- student bounding boxes
- anonymous track IDs
- seating location
- current status
- current confidence
- event badge

The system must visibly demonstrate that rear-row students are actually being tracked.

---

# 29. VISIBILITY AND COVERAGE METRICS

Create a live coverage metric:

```text
Students expected: 58
Students detected: 57
Students tracked: 56
Low visibility: 2
Occluded: 3
```

Also:

```text
Coverage = tracked_students / expected_students
```

Display:

`97% monitoring coverage`

Do not silently claim complete coverage when 5 students are not visible.

---

# 30. OCCLUSION HANDLING

Implement explicit visibility states:

```text
VISIBLE
PARTIALLY_OCCLUDED
HEAVILY_OCCLUDED
TEMPORARILY_LOST
OUT_OF_FRAME
```

When a student's body is temporarily hidden:

- preserve track ID
- extrapolate motion
- avoid generating unreliable events
- reacquire using appearance + spatial context

Never treat absence of evidence as evidence of cheating.

---

# 31. SEAT-BASED TRACK PERSISTENCE

Allow the proctor to define expected student seats.

Use seat regions to stabilize tracking.

For example:

```text
Seat R3-C4
expected track: StudentTrack_019
```

If a student leaves a seat temporarily:

- retain identity
- update position
- do not generate false track swaps

If track identity uncertainty becomes high:

`TRACK_UNCERTAIN`

must be surfaced.

---

# 32. CAMERA CALIBRATION

Add optional camera calibration.

For fixed CCTV:

- lens distortion
- field of view
- perspective
- homography
- approximate floor plane

Use calibration to improve:

- inter-student distances
- desk relationships
- row-aware object scaling
- spatial interaction reasoning

If multiple cameras exist, support synchronized multi-camera architecture.

Modern multi-view systems can use calibrated cameras to maintain cross-camera target identity and spatial consistency. NVIDIA's current multi-view tracking documentation explicitly emphasizes camera calibration and cross-camera identity re-association.

---

# 33. OPTIONAL MULTI-CAMERA SUPPORT

Design the backend so it can later accept:

```text
Camera 1
Camera 2
Camera 3
...
```

Each camera should have:

- calibration
- local tracker
- camera ID
- frame timestamp

Then perform:

```text
camera-local tracking
       ↓
global identity association
       ↓
global student track
```

Do not hardwire the architecture to one camera.

---

# 34. MODEL ROUTING

Create a model orchestration layer.

Example:

```python
PerceptionRouter
```

Responsibilities:

- determine which model is needed
- select input resolution
- choose crop
- decide whether refinement is needed
- batch inference where possible
- manage GPU memory
- cache model instances

Example:

```text
NORMAL FRAME
→ cheap inference

SMALL OBJECT FOUND
→ crop refinement

SUSPICIOUS INTERACTION
→ temporal analysis

HIGH-CONFIDENCE EVENT
→ expensive segmentation / evidence generation
```

Do not run the most expensive model on every frame unnecessarily.

---

# 35. GPU BATCHING

The report identifies sequential frame processing as a current performance limitation.

Implement:

- batch frame inference
- batched student crops
- asynchronous pipeline stages
- producer/consumer queues
- GPU inference workers
- CPU preprocessing workers
- asynchronous evidence generation

Architecture:

```text
Video Decoder
      ↓
Frame Queue
      ↓
Detection Workers
      ↓
Tracking
      ↓
Student Crop Queue
      ↓
Specialized Detection
      ↓
Temporal Engine
      ↓
Evidence Queue
      ↓
Database
```

Avoid blocking the FastAPI request thread.

---

# 36. ANALYSIS JOB SYSTEM

Replace:

```text
POST /analyze
→ run entire video synchronously
```

with a background job model.

Example:

```text
POST /analysis
→ job_id

GET /analysis/{job_id}
→ status

GET /analysis/{job_id}/progress
→ 63%

GET /analysis/{job_id}/events
→ partial results
```

States:

```text
QUEUED
INITIALIZING
DETECTING
TRACKING
ANALYZING
GENERATING_EVIDENCE
COMPLETED
FAILED
CANCELLED
```

Frontend must show real progress.

---

# 37. DATABASE

The existing project currently stores JSON on the filesystem rather than using persistent database storage.

Implement SQLite for local deployment and PostgreSQL-compatible schemas for production.

Suggested tables:

```text
videos
analysis_jobs
students
student_tracks
seats
detections
objects
poses
behavior_windows
events
event_evidence
model_runs
metrics
```

Keep raw video/evidence files in object/file storage.

Do not put large videos inside relational DB blobs.

---

# 38. STRICT EVENT TAXONOMY

Define:

```text
NORMAL
LOOK_AROUND
TURNING_AROUND
PHONE_USAGE
CHEAT_CHIT
UNAUTHORIZED_MATERIAL
PAPER_COPYING
NOTE_PASSING
OBJECT_PASSING
SUSPICIOUS_HAND_INTERACTION
POSSIBLE_COMMUNICATION
TRACK_UNCERTAIN
OCCLUDED
```

Every event must contain:

```text
event_id
track_id
type
severity
confidence
start_time
end_time
duration
evidence
supporting_signals
contradicting_signals
model_version
```

---

# 39. SUPPORTING VS CONTRADICTING EVIDENCE

This is extremely important.

Do not merely calculate positive evidence.

Store both.

Example:

```json
{
  "supporting_signals": [
      "phone visible",
      "right hand touching phone",
      "head directed toward phone",
      "interaction lasted 7.2 sec"
  ],
  "contradicting_signals": [
      "phone confidence dropped below threshold twice"
  ]
}
```

This makes the system explainable.

---

# 40. DO NOT OVER-FLAG NORMAL BEHAVIOR

Build a hard-negative evaluation suite.

Include:

- looking at clock
- looking at proctor
- turning to hear something
- stretching
- coughing
- drinking water
- adjusting chair
- picking up pen
- dropping pen
- scratching face
- touching hair
- looking down
- changing posture
- writing normally
- reaching for own stationery
- talking to proctor

The system should learn:

```text
suspicious visual behavior
≠
malpractice
```

Only persistent contextual evidence should create a high-severity malpractice event.

---

# 41. TRAINING DATA MUST BE CLASSROOM-SPECIFIC

Create a dataset with realistic classroom footage.

Do not benchmark only on public generic person/phone datasets.

Split by:

```text
classroom
camera
student population
recording session
```

Never randomly place adjacent video frames into both train and test datasets because that causes leakage.

The correct split is session-level:

```text
Session A → train
Session B → train
Session C → validation
Session D → test
```

---

# 42. REQUIRED DATASET METRICS

For detection:

- precision
- recall
- mAP50
- mAP50-95

For tracking:

- HOTA
- IDF1
- MOTA
- ID switches
- fragmentation

For behavior recognition:

- precision
- recall
- F1
- confusion matrix
- per-class recall

For event systems:

- false positives per hour
- false positives per student-hour
- missed events per hour
- average detection latency
- event duration error

Most importantly:

## Student coverage

Measure:

```text
% of visible students successfully tracked
```

and:

```text
% of student-minutes with valid monitoring
```

---

# 43. DEFINE ACCEPTANCE TARGETS

Do not mark the implementation complete merely because automated tests pass.

Create an explicit benchmark.

Example targets — treat these as engineering goals that must be validated on an unseen evaluation set, not as claims about what the model already achieves:

```text
Student detection recall: >= 97%
Student track continuity: >= 95%
IDF1: >= 90%
Rear-row detection recall: >= 95%
Phone detection recall on student crops: >= 95%
Cheat-chit detection recall: >= 90%
Paper-copying precision: >= 90%
Paper-copying recall: >= 85%
Phone-usage precision: >= 95%
Turning-around precision: >= 90%
Overall event false-positive rate: <= 0.5/student-hour
```

If the actual model cannot achieve these values, report the measured values honestly and show where the gap exists.

Never fabricate performance.

---

# 44. REAR-ROW BENCHMARK

Create a dedicated test suite:

```text
foreground
middle
rear
corner
partially occluded
small student
small object
poor lighting
```

Report metrics by region.

Example:

```text
Foreground recall: 99.1%
Middle recall:      98.2%
Rear recall:        94.7%
Corner recall:      92.5%
```

This is much more useful than one global accuracy value.

---

# 45. CONFUSION MATRIX FOR MALPRACTICE

Generate:

```text
                   predicted
             phone copying chit turn normal
actual phone
actual copying
actual chit
actual turn
actual normal
```

Look especially for:

- normal → cheating false positives
- turning → copying confusion
- looking around → copying confusion
- phone → generic object confusion
- paper → cheat chit confusion

---

# 46. MODEL VERSIONING

Every analysis must record:

```text
detector_model
pose_model
tracker
temporal_model
segmentation_model
model_checksum
configuration_version
dataset_version
```

This is mandatory for reproducibility.

---

# 47. CONFIGURATION

Move all important values to YAML or environment-driven configuration.

Do not hardcode:

- confidence
- image size
- tile count
- tile overlap
- inference FPS
- tracking thresholds
- temporal duration
- event thresholds
- model paths
- GPU selection
- evidence duration
- database path
- storage directory

Use something such as:

```yaml
perception:
  full_frame_imgsz: 1280
  tile_imgsz: 1280
  tile_overlap: 0.20

tracking:
  tracker: botsort
  reid: true
  track_buffer: 60

temporal:
  window_seconds: 8
  stride_seconds: 0.5

events:
  phone:
    min_duration_seconds: 2.0
  paper_copying:
    min_duration_seconds: 2.5
  turning:
    min_duration_seconds: 2.0
```

All values must remain tunable.

---

# 48. TESTING

Create a serious automated test suite.

Test:

- frame decoding
- tiling
- coordinate remapping
- detection merging
- student tracking
- track persistence
- track loss/recovery
- ReID
- pose extraction
- object attribution
- student crop generation
- temporal windows
- event aggregation
- confidence calibration
- evidence generation
- database persistence
- API jobs
- export
- frontend rendering

Add synthetic tests.

Example:

```text
Two students
One phone
One object transfer
One head turn
Known ground truth
```

Verify expected event behavior.

---

# 49. END-TO-END TEST VIDEO

Create or include benchmark scenarios:

### Scenario A
50 students, normal writing.

Expected:

```text
0 high-confidence cheating events
```

### Scenario B
One student uses phone.

Expected:

```text
phone_usage → StudentTrack_X
```

### Scenario C
One student repeatedly looks at neighboring answer sheet.

Expected:

```text
paper_copying → StudentTrack_X
```

### Scenario D
One student has a cheat chit.

Expected:

```text
cheat_chit → StudentTrack_X
```

### Scenario E
One student passes a chit.

Expected:

```text
note_passing → StudentTrack_X + StudentTrack_Y
```

### Scenario F
One student simply turns around once.

Expected:

```text
turning_around
NOT confirmed malpractice
```

### Scenario G
Five students simultaneously perform different behaviors.

The system must detect them independently.

---

# 50. SIMULTANEOUS EVENT TEST

This is mandatory.

Create a test in which:

```text
Student 5 → phone
Student 13 → paper copying
Student 21 → chit
Student 29 → turns around
Student 37 → note passing
Student 44 → normal
```

All happen within approximately the same period.

The system must not process only the first suspicious student.

All student tracks must continue simultaneously.

---

# 51. FRONTEND REBUILD

Keep the existing React dashboard, but transform it into a professional proctoring dashboard.

Add:

### Classroom View

Live/grid overview of all students.

### Student View

Detailed per-student analysis.

### Event View

Timeline of all events.

### Evidence View

Video clip + annotated frame + explanation.

### Confidence View

Confidence over time.

### Coverage View

Which students are visible/tracked.

### Model Diagnostics

FPS
GPU utilization
latency
model confidence
track count
dropped frames

Do not expose excessive technical information by default; place diagnostics behind an advanced panel.

---

# 52. EVENT REVIEW PANEL

Clicking an event should show:

```text
EVENT
Paper Copying

STUDENT
StudentTrack_023

DURATION
6.5 seconds

CONFIDENCE
91%

TARGET
StudentTrack_024

WHY FLAGGED
• repeated leftward gaze
• head direction aligned with neighbor's paper
• sustained interaction for 6.5 seconds
• 7 repeated glances
• target student remained within relevant spatial region

COUNTER-EVIDENCE
• partial occlusion for 0.8 sec
```

This dramatically improves proctor trust.

---

# 53. HUMAN-IN-THE-LOOP

The system must not automatically punish students.

Provide:

```text
CONFIRM
DISMISS
MARK INCORRECT
NOT SURE
```

Store reviewer decisions.

Create a feedback dataset.

This can later be used to improve the model.

A dismissed event should become a hard-negative candidate for future training.

---

# 54. ACTIVE LEARNING PIPELINE

Create:

```text
high uncertainty events
        ↓
proctor review
        ↓
correct label
        ↓
dataset
        ↓
retraining
        ↓
evaluation
```

Prioritize samples where:

- confidence ~0.5
- track uncertainty is high
- model disagreement exists
- multiple classifiers disagree
- proctor frequently dismisses the class

---

# 55. MODEL DISAGREEMENT

Where practical, use independent signals.

Example:

```text
Pose model:
looks left = TRUE

Object interaction model:
neighbor paper interaction = TRUE

Temporal model:
copying probability = 0.87

Rule engine:
duration sufficient = TRUE
```

Then combine evidence.

If:

```text
Pose = TRUE
Temporal = LOW
Object = FALSE
```

do not produce a high-confidence copying event.

This is significantly safer than one heuristic generating the accusation.

---

# 56. UNCERTAINTY HANDLING

Create three final levels:

```text
OBSERVATION
SUSPICIOUS
LIKELY MALPRACTICE
```

Example:

`LOOK_AROUND — 62%`

should remain:

`Suspicious behavior`

rather than:

`Cheating detected`

Only sufficiently supported events should become:

`Likely malpractice`

---

# 57. PERFORMANCE ENGINEERING

Benchmark:

- CPU usage
- GPU usage
- RAM
- VRAM
- FPS
- latency
- throughput

Support:

- CPU fallback
- CUDA
- mixed precision
- FP16
- TensorRT/ONNX where appropriate

Ultralytics currently supports deployment through ONNX, TensorRT, OpenVINO and other backends, so design model loading/export around the deployment target rather than tying inference permanently to Python/PyTorch.

---

# 58. RESOURCE-AWARE MODEL SELECTION

Implement profiles.

## Accuracy Profile

Large detector
Large pose model
high-resolution tiles
segmentation refinement
strongest tracker

## Balanced Profile

Medium detector
medium pose
moderate tile count
selective refinement

## Real-Time Profile

Small detector
efficient tracker
adaptive crops
limited refinement

The UI should clearly show:

`Analysis Profile: Accuracy`

---

# 59. VIDEO QUALITY ANALYSIS

Before inference, inspect:

- resolution
- FPS
- bitrate
- blur
- illumination
- compression
- camera shake
- field of view

Generate:

```text
VIDEO QUALITY
★★★★☆
```

and explain limitations.

For example:

```text
Rear-row phone detection may be unreliable because the
camera resolution provides fewer than N pixels across the
typical phone object.
```

Never pretend model architecture can recover information that is absent from the video.

---

# 60. AUTOMATIC QUALITY GATING

If the camera is too poor:

```text
Insufficient visual resolution for reliable rear-row object detection.
```

Do not silently generate misleading results.

---

# 61. SECURITY

Fix the current permissive production configuration.

The report currently notes:

```python
allow_origins=["*"]
```

in CORS.

Replace this with configurable trusted origins.

Also implement:

- authentication-ready API structure
- rate limiting
- upload validation
- secure file names
- path traversal protection
- maximum video duration
- maximum resolution
- resource quotas
- safe temporary files
- cleanup jobs
- structured audit logs

---

# 62. PRIVACY

Do not introduce facial recognition merely to track students.

Use anonymous track IDs.

If faces appear in exported evidence, treat them as sensitive evidence and implement appropriate:

- access controls
- retention policies
- deletion
- audit logging

Do not permanently store unnecessary biometric information.

---

# 63. OBSERVABILITY

Implement structured logs.

Example:

```text
analysis_id
video_id
frame
timestamp
track_count
detections
events
gpu_latency
model_latency
queue_depth
```

Add metrics.

Example:

```text
frames_processed
frames_dropped
tracks_active
tracks_lost
id_switches
events_generated
events_reviewed
events_dismissed
```

---

# 64. DEBUG MODE

Create a developer debug mode.

Display:

- raw detector output
- tracker IDs
- keypoints
- crops
- object associations
- temporal windows
- event score components

This should be toggleable and never necessary for normal proctor usage.

---

# 65. EVIDENCE QUALITY

Evidence images must show:

- student ID
- event type
- timestamp
- confidence
- relevant body region
- relevant object
- target student if interaction-based

For paper copying, explicitly draw:

```text
Student A
     ↓ gaze/head direction
     ↘
      Student B's paper
```

For phone usage:

```text
Student
  ↓
Hand
  ↓
Phone
```

For note passing:

```text
Student A → Object → Student B
```

This is much more useful than simply drawing a red rectangle around a student.

---

# 66. FILE STRUCTURE TARGET

Refactor toward something approximately like:

```text
backend/
    api/
    config/
    database/
    ingestion/
    pipeline/
    perception/
        detection/
        tracking/
        pose/
        segmentation/
        crops/
    behavior/
        gaze/
        posture/
        interaction/
        temporal/
    events/
    evidence/
    jobs/
    metrics/
    storage/
    schemas/

frontend/
    components/
    pages/
    classroom/
    students/
    events/
    evidence/
    diagnostics/
    timeline/

datasets/
    annotations/
    scripts/
    benchmark/

models/
    configs/

tests/
    unit/
    integration/
    benchmark/
    e2e/
```

Do not blindly create this structure if the existing repository has better equivalents; preserve sound existing abstractions where possible.

---

# 67. API TARGET

Create clean APIs such as:

```text
POST /videos/upload

POST /analysis

GET /analysis/{id}

GET /analysis/{id}/progress

GET /analysis/{id}/students

GET /analysis/{id}/students/{track_id}

GET /analysis/{id}/events

GET /analysis/{id}/events/{event_id}

GET /analysis/{id}/evidence/{event_id}

GET /analysis/{id}/coverage

GET /analysis/{id}/metrics

POST /events/{event_id}/review

GET /export/{analysis_id}/json

GET /export/{analysis_id}/csv

GET /export/{analysis_id}/zip
```

Maintain backwards compatibility where practical.

---

# 68. FRONTEND EVENT FILTERING

Allow filtering by:

```text
All
Phone
Paper Copying
Cheat Chit
Note Passing
Turning Around
Looking Around
Unauthorized Material
Communication
Uncertain
```

Also filter by:

```text
confidence
student
seat
timestamp
severity
```

---

# 69. SEARCH

Implement:

```text
Search StudentTrack_023
```

Then show:

- all events
- evidence
- timeline
- confidence
- track coverage
- seat

---

# 70. REPORT GENERATION

The existing JSON/CSV/ZIP export functionality should remain, but improve it to include:

- event details
- evidence
- confidence
- model versions
- analysis profile
- coverage
- performance
- reviewer decisions

The system should generate a professional final report.

---

# 71. DO NOT FAKE RESULTS

This rule is absolute.

Never:

- hardcode a cheating event
- generate synthetic detections pretending to come from the model
- assume every phone is cheating
- assume every head turn is copying
- assign arbitrary confidence values
- mark the project “100% accurate” without benchmark evidence

If something does not work, report it.

---

# 72. IMPLEMENTATION ORDER

Follow this order:

### PHASE 1
Audit and refactor current pipeline.

### PHASE 2
Implement new configuration system.

### PHASE 3
Implement student-first tracking.

### PHASE 4
Implement tiling + perspective-aware inference.

### PHASE 5
Implement student-centric crops.

### PHASE 6
Upgrade detector and pose models.

### PHASE 7
Implement custom object taxonomy.

### PHASE 8
Implement temporal feature engine.

### PHASE 9
Implement malpractice classifiers.

### PHASE 10
Implement evidence/event architecture.

### PHASE 11
Implement job processing and asynchronous inference.

### PHASE 12
Implement database persistence.

### PHASE 13
Upgrade frontend.

### PHASE 14
Implement benchmark suite.

### PHASE 15
Run complete evaluation.

### PHASE 16
Fix every major failure revealed by evaluation.

Do not stop after implementing features.

---

# 73. REQUIRED FINAL VALIDATION

At the end, run:

```bash
pytest -q
```

and every available frontend test/lint/build command.

Also run:

```bash
npm run build
npm run lint
```

Create a benchmark command such as:

```bash
python -m benchmark.evaluate --video <test_video>
```

Generate a machine-readable result:

```json
{
  "student_detection_recall": ...,
  "tracking_idf1": ...,
  "rear_row_recall": ...,
  "phone_precision": ...,
  "phone_recall": ...,
  "paper_copying_precision": ...,
  "paper_copying_recall": ...,
  "false_positive_rate": ...,
  "processing_fps": ...
}
```

---

# 74. FINAL SUCCESS CRITERIA

Do NOT declare the project complete until all of the following are true:

1. Approximately 50–60 students can be tracked in a realistic classroom video.

2. Rear-row students are explicitly benchmarked and no longer silently ignored.

3. Students retain stable anonymous IDs across frames.

4. Small-object detection works through student-centric high-resolution crops.

5. Generic COCO object recognition is not the sole mechanism for detecting cheating artifacts.

6. Phone usage is classified using object + hand + gaze + temporal evidence.

7. Paper copying is classified using target-paper interaction and temporal evidence rather than simple head-yaw raycasting.

8. Cheat chits are explicitly detected as their own object/behavior category.

9. Note passing is based on actual object/hand transfer patterns rather than student-centroid distance alone.

10. Turning around is treated separately from paper copying.

11. Looking around is not automatically classified as cheating.

12. Normal classroom behaviors are included as hard negatives.

13. Every flag contains evidence.

14. Every flag has calibrated confidence.

15. Every event can be reviewed by a human proctor.

16. Detection coverage is visible to the user.

17. The system explicitly reports when visual quality is insufficient.

18. The pipeline supports GPU batching and asynchronous processing.

19. The backend no longer blocks the API while a long video is processed.

20. Database persistence is implemented.

21. All model versions/configuration versions are recorded.

22. Full automated tests pass.

23. Frontend builds successfully.

24. Benchmark metrics are generated from real evaluation data.

25. No accuracy claims are made without measured evidence.

---

# 75. CRITICAL ENGINEERING PRINCIPLE

The final architecture should conceptually be:

```text
                    CCTV VIDEO
                         │
                         ▼
                VIDEO QUALITY CHECK
                         │
                         ▼
              CLASSROOM CALIBRATION
                         │
                         ▼
               FULL-FRAME DETECTION
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      SPATIAL TILING          PERSON TRACKING
              │                     │
              └──────────┬──────────┘
                         ▼
                STUDENT ASSOCIATION
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          POSE       STUDENT      OBJECTS
        ESTIMATION     CROPS      DETECTION
             │           │           │
             └───────────┼───────────┘
                         ▼
                HAND / HEAD / DESK
                   RELATIONSHIPS
                         │
                         ▼
                 TEMPORAL FEATURES
                         │
                         ▼
                BEHAVIOR CLASSIFIERS
                         │
                         ▼
              MULTI-SIGNAL FUSION
                         │
                         ▼
                CONFIDENCE CALIBRATION
                         │
                         ▼
                  EVENT ENGINE
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        EVIDENCE               PROCTOR REVIEW
             │                       │
             └───────────┬───────────┘
                         ▼
                   FINAL REPORT
```

The key principle is:

**Detect the student first. Track the student continuously. Analyze the student's local high-resolution region. Understand actions over time. Combine multiple independent signals. Only then classify malpractice.**

Do not attempt to solve a 50–60 student classroom by running a generic object detector on the full CCTV image and attaching a few geometric rules afterward.

---

# 76. WHAT YOU MUST RETURN AFTER IMPLEMENTATION

When finished, provide:

1. Complete modified file tree.

2. Detailed summary of every changed file.

3. Explanation of the new architecture.

4. Model choices and reasons.

5. Detection pipeline explanation.

6. Tracking pipeline explanation.

7. Rear-row detection strategy.

8. Each malpractice classifier's logic.

9. Temporal-analysis methodology.

10. Database schema.

11. API changes.

12. Frontend changes.

13. Configuration changes.

14. Testing results.

15. Benchmark results.

16. Known limitations.

17. Exact commands to start the system.

18. Exact commands to run benchmarks.

19. Exact commands to train/fine-tune custom models.

20. A section clearly distinguishing:
   - implemented
   - tested
   - benchmarked
   - partially working
   - not yet reliable

Do not provide a superficial summary.

The final repository must be executable and testable, not merely architecturally impressive.