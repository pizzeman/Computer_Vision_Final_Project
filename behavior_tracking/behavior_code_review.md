# Code Review: behavior.py

Scope focus: logical errors, label handling, and type/array-size correctness.

## High Severity Findings

1. **File currently does not parse (syntax error)**
   - **Location:** `behavior_tracking/behavior.py:237`
   - There is an extra trailing fragment in this line:
     - `... in test_video_as_frames]_test_video_as_frames]`
   - This causes an immediate `SyntaxError: unmatched ']'`, so nothing in this module can run.

2. **Wrong `time` usage causes runtime failure in `run()`**
   - **Location:** `behavior_tracking/behavior.py:2`, `behavior_tracking/behavior.py:227`
   - `from time import time` imports the function, but code uses `time.time()` at line 227.
   - This would raise `AttributeError: 'builtin_function_or_method' object has no attribute 'time'` (once syntax is fixed).

3. **Training data structure is incompatible with `train_model()` expectations**
   - **Location:** `behavior_tracking/behavior.py:172-189`, `behavior_tracking/behavior.py:240-242`
   - `train_model()` expects each item to be a tuple-like sample where:
     - `item[0]` is frames
     - `item[2]` is label
   - But `train_videos_normalized` is built as a list of arrays only (no labels), so:
     - `item[0]` becomes the first frame in the sequence, not the sample tensor
     - `item[2]` becomes a frame slice, not a class label
   - Result: label and feature extraction are logically wrong and break downstream shape assumptions.

4. **Model input dimensionality mismatch (4D tensors passed into 1D temporal CNN expecting 3D)**
   - **Location:** `behavior_tracking/behavior.py:147-149`, `behavior_tracking/behavior.py:188-192`, `behavior_tracking/behavior.py:219-222`
   - `normalize_frames()` outputs shape `(T, L, F)`, so batch is `(B, T, L, F)`.
   - `BehaviorCNN.forward()` expects `(B, T, F)` and calls `x.permute(0, 2, 1)` (3 dims).
   - Passing 4D tensors will fail at permutation/conv input stage.

5. **Label handling is inconsistent with `CrossEntropyLoss` and metrics**
   - **Location:** `behavior_tracking/behavior.py:176`, `behavior_tracking/behavior.py:186-189`, `behavior_tracking/behavior.py:197`, `behavior_tracking/behavior.py:202-205`, `behavior_tracking/behavior.py:220-223`
   - `CrossEntropyLoss` expects class indices of shape `(B,)` (`torch.long`) or class probabilities (`float`) with special semantics.
   - Code constructs labels from `item[2]` and casts to `torch.long`, then computes `labels_tensor.argmax(dim=1)`.
   - This suggests one-hot assumptions, but one-hot labels are never built into dataset samples. Current label path is broken.

6. **Feature concatenation in `normalize_frames()` has incompatible array sizes**
   - **Location:** `behavior_tracking/behavior.py:118-119`
   - `frames` shape before concat: `(T, L, 4)`.
   - `onehot[:, None, :]` shape: `(T, 1, M)`.
   - Concatenating along axis 2 requires matching axis 1 (`L` vs `1`), which fails unless `L == 1`.
   - This is a direct array-size bug in metadata feature appending.

## Medium Severity Findings

7. **Potential frame-file discovery bug due to formatted index containing spaces**
   - **Location:** `behavior_tracking/behavior.py:62`
   - Filename uses `i:6d`, producing left-padded spaces (e.g., `"     0"`), likely not matching real file names.
   - If files are named with zero padding/no padding, this yields empty frame lists.

8. **Empty-frame path not handled before indexing and velocity logic**
   - **Location:** `behavior_tracking/behavior.py:89-92`, `behavior_tracking/behavior.py:99`, `behavior_tracking/behavior.py:111`
   - If `frames` is empty, code hits `frames[-1]` and fails.
   - Even apart from padding, center/velocity assumptions require non-empty arrays.

9. **Hard-coded landmark indices without shape validation**
   - **Location:** `behavior_tracking/behavior.py:99`, `behavior_tracking/behavior.py:103-104`
   - Uses landmarks `19`, `0`, and `16` with no guard on `L`.
   - If inferred keypoint count is smaller than expected, this throws index errors.

10. **Data leakage in split logic (validation contains test subset)**
    - **Location:** `behavior_tracking/behavior.py:49-55`, `behavior_tracking/behavior.py:229`, `behavior_tracking/behavior.py:233-234`
    - `validation_test_ids` includes first `n_validation + n_test` samples.
    - `test_video_ids` is a subset of that same range.
    - Validation therefore overlaps with test, contaminating evaluation.

11. **Metrics aggregation code is invalid and semantically incorrect**
    - **Location:** `behavior_tracking/behavior.py:207-210`
    - `train_acc_history / len(train_data)` and `val_acc_history / len(validation_data)` divide Python lists by ints (TypeError).
    - Even if fixed syntactically, denominator should match number of recorded metric points (batches/epochs), not dataset length.

12. **Checkpoint path likely inconsistent with repository structure**
    - **Location:** `behavior_tracking/behavior.py:75`
    - Uses `./results/hrnet_best.pth` while workspace tree shows weights under `limb_tracking/results/hrnet_best.pth`.
    - Likely file-not-found depending on process working directory.

## Low Severity / Maintainability

13. **Duplicate assignment and typo reduce clarity and increase error risk**
    - **Location:** `behavior_tracking/behavior.py:235`, `behavior_tracking/behavior.py:237-238`
    - `train_vidoes_as_limbs` typo is used consistently but is easy to misuse.
    - `test_videos_as_limbs` is assigned twice (second assignment overwrites first).

14. **Unused argument in `run_limb_tracking()`**
    - **Location:** `behavior_tracking/behavior.py:69`
    - `animal` parameter is never used.

## Open Questions / Assumptions

1. Is the intended classifier target **behavior only** (`num_classes = len(BEHAVIORS)`) or `(animal, behavior)` pair? Current logic is ambiguous because both are one-hot encoded into features while only behavior is used for class count.
2. Should limb dimension `L` be flattened into features (`L * F`) per timestep, pooled over landmarks, or modeled with a spatial-temporal architecture? Current network design implies no explicit landmark axis.
3. Is `split_video_ids()` intentionally deterministic (no shuffle/stratification)? If metadata is ordered, split quality may be biased.

## Suggested Priority Fix Order

1. Fix syntax error and `time` import usage so script can execute.
2. Define one canonical sample schema (e.g., `(frames, animal, behavior_idx)` or dict) and apply it consistently through preprocess/train/val/test.
3. Resolve model input contract: either reshape to `(B, T, F_total)` or redesign model for `(B, T, L, F)`.
4. Correct label pipeline to use class indices for `CrossEntropyLoss`.
5. Fix `normalize_frames()` metadata concatenation and add empty/shape guards.
6. Correct train/val/test split leakage.
7. Fix metrics aggregation and checkpoint/file path assumptions.

## Validation Performed

- Syntax check command: `python3 -m py_compile behavior_tracking/behavior.py`
- Result: fails with `SyntaxError` at line 237.
