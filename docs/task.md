# IPL Data Acquisition Project

## Current Objective
Get IPL ball-by-ball data (2020-2024) from Cricsheet.org as an open source alternative to scraping ESPN CricInfo.

---

## Tasks

### Phase 1: Planning & Research ✅
- [x] Research data sources (ESPN CricInfo, Cricsheet.org, Kaggle)
- [x] Identify Cricsheet.org as best option (free, structured, legal)
- [x] Create implementation plan
- [x] Get user approval

### Phase 2: Data Acquisition ✅
- [x] Create download script (`download_ipl_data.py`)
- [x] Download IPL JSON data from Cricsheet.org (1,169 matches)
- [x] Extract to `data/raw/`

### Phase 3: Data Processing ✅
- [x] Create processing script (`process_ipl_data.py`)
- [x] Filter for 2020-2024 seasons (279 matches)
- [x] Generate `deliveries.csv` (67,303 rows)
- [x] Generate `players.csv` (336 players with stats)
- [x] Generate `venues.csv` (20 venues)

### Phase 4: ML Model Development ✅
- [x] Design model architecture and features
- [x] Create training data pipeline
- [x] Build ball outcome prediction model
- [x] Build player value model
- [x] Train and evaluate models (XGBoost, LightGBM, Random Forest)
- [x] Export for JavaScript integration
- [x] Integrate with game simulation

### Phase 5: Future Enhancements (GPU/Deep Learning) 📋
- [ ] Set up GPU environment (CUDA/M1 Metal)
- [ ] Train neural network (MLP) for comparison
- [ ] Train LSTM for sequence modeling (ball-by-ball patterns)
- [ ] Try player/venue embeddings
- [ ] Ensemble model (combine GB + DL)
- [ ] Compare all models and select best

### Phase 5: Deep Learning Enhancements ✅
- [x] Set up GPU environment (CUDA)
- [x] Train neural network (MLP) - 41.31% accuracy
- [x] Train Deep MLP - **41.44% accuracy** 🏆
- [x] Compare models and select best
- [x] Export for JavaScript integration