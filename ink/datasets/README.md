# ink/datasets

Every ink training dataset lives in the library's `datasets/`, one folder per kind, each built by one file. `dataset_check.py` holds the schema; `hand_mark_strokes.py` carries marks to piece pixels and scan voxels; `text_evidence.py` keeps only published maps that hold rows of writing.

```text
labelled_cases/CASE/            case.json, layers/ (brick store), ink.npy, supervised.npy[, registration.json]; manifest.jsonl lists them   labelled_cases.py
labelled_case_stacks/CASE.npy   a case's layers inside its box, derived                         ../training/model_training.py
infrared_fragments/FragN/       ir.png, mask.png, inklabels.png                                 scans/transfer/fetch_fragment.py
published_pairs/SCROLL/         SEGMENT-Y-X.npy (9 µm stack), .png (published map), .json (face layers); maps/, plan.json; published_pairs_writing/ planned for maps with rows of writing, not yet built   published_pairs.py
own_mesh_labels/ROUTE/SCROLL/   CASE-PIECE.npy, -labels.npz, .json; reads/ a reader's own reads (own_reads.py); ROUTE/census/BAND.json   own_mesh_labels.py
letter_spots/ROUTE/SCROLL/      candidates.json; UNIT-piece-N-Y-X.npy (17 layers), .json, reads/ (second reader), shown/ (first read)   letter_spots.py
target_stacks/SCROLL/UNIT/      piece-NNNN.npy: the target's stacks, used as blank or background   target_stacks.py
agreed_targets/KIND/NAME.npy    a byte a pixel (0 unknown, 1 blank, 2 ink) where hand label and fine map agree; KIND as the source folder   agreed_targets.py
depth_mean_targets/KIND/NAME.npy   a reader's mean read over depths -4..+4 of a taught window, as bytes, for a single-read student   depth_mean_targets.py
hand_marks/ROUTE/SCROLL/SCAN/   marks.jsonl, layouts/DIGEST.json; two_unwraps/SCROLL/ the same   website/site_server.py
eye_letters/                    letters.npz (ink, own, any, known: one 144 px window a letter) and letters.json (letter, case, partition, AUC): the 577 letters named by eye, on published ink maps or infrared   eye_letters.py

./run.sh ink.datasets.dataset_check     every dataset against the schema; writes datasets/manifest.json, exit 1 on any fault
./run.sh ink.datasets.labelled_cases | published_pairs | own_mesh_labels census|build|published BAND [ROUTE] | own_reads SCROLL WEIGHTS
./run.sh ink.datasets.stitched_windows BAND OUT [CASE...]     a level band's read blocks stitched back per source piece around each labelled case, one stack, labels carried raw and readable
./run.sh ink.datasets.letter_spots candidates SCROLL SCAN | build SCROLL SCAN WEIGHTS     needs the read's kept stacks
./run.sh ink.datasets.agreed_targets build | depth_mean_targets WEIGHTS
./run.sh ink.datasets.eye_letters build     needs the atlas build's letter bodies (HERCULEX_ATLAS_WORK, default library/atlas/work)
```
CASE is SCROLL-SEGMENT-VOXEL-DISTANCE-ENERGY. `case.json` holds case_id, sample, voxel_um, box, truth and partition (development or holdout, fixed by a hash of scroll and segment).
A mark is one JSON line: id, kind (ink, blank, spot), width_mm, layer, points in sheet mm, layout digest. A removal names an id. The log is append-only, and the labels are the marks that have not been removed.
