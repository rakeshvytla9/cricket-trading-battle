# Cricket Trading Battle - ML Model Walkthrough

## Summary

Built and integrated an **ML-powered ball outcome prediction system** trained on 67,303 real IPL deliveries (2020-2024).

---

## Models Trained

| Model | Accuracy | Log Loss | Training Time | Winner |
|-------|----------|----------|---------------|--------|
| **LightGBM** | 41.33% | 1.448 | 0.76s | 🏆 |
| Random Forest | 41.01% | 1.452 | 1.42s | |
| XGBoost | 40.63% | 1.466 | 2.64s | |

---

## Files Created

| File | Purpose |
|------|---------|
| `data/create_training_data.py` | Feature engineering (21 features) |
| `ml/train_advanced_models.py` | XGBoost/LightGBM/RF training |
| `ml/ball_outcome_model.json` | LightGBM model weights |
| `ml_predictor.js` | JavaScript prediction engine |
| `data/player_stats.json` | 50 player statistics |
| `data/matchup_table.json` | 2,297 matchups |

---

## Demo Recording

See `docs/lightgbm_game_test_1767158476162.webp`

---

## Screenshot

See `docs/match_final_test_1767158782931.png`

---

## Future Plans (Phase 5 - GPU)

- [ ] Train neural network (MLP) for comparison
- [ ] Train LSTM for sequence modeling
- [ ] Try player/venue embeddings
- [ ] Ensemble model (combine GB + DL)
- [ ] Compare all models and select best
