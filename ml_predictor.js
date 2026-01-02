/**
 * Cricket ML Predictor
 * Uses trained model to predict ball outcomes with realistic probabilities
 */

class CricketMLPredictor {
    constructor() {
        this.model = null;
        this.playerStats = null;
        this.matchupTable = null;
        this.loaded = false;

        // Outcome classes
        this.OUTCOMES = ['DOT', 'SINGLE', 'DOUBLE', 'TRIPLE', 'FOUR', 'SIX', 'WICKET'];

        // Feature order (must match training)
        this.FEATURE_NAMES = [
            'over', 'ball_in_over', 'innings', 'phase_powerplay', 'phase_middle', 'phase_death',
            'runs_scored', 'wickets_lost', 'run_rate', 'required_rate', 'balls_remaining',
            'batsman_sr', 'batsman_avg', 'batsman_experience',
            'bowler_economy', 'bowler_sr', 'bowler_experience',
            'matchup_sr', 'matchup_balls', 'venue_avg_score', 'recent_form'
        ];
    }

    /**
     * Load model and data files
     */
    async load() {
        try {
            // Load model
            const modelResponse = await fetch('./ml/ball_outcome_model.json');
            this.model = await modelResponse.json();

            // Load player stats
            const playerResponse = await fetch('./data/player_stats.json');
            this.playerStats = await playerResponse.json();

            // Load matchup table
            const matchupResponse = await fetch('./data/matchup_table.json');
            this.matchupTable = await matchupResponse.json();

            this.loaded = true;
            console.log('✅ ML Predictor loaded successfully');
            console.log(`   - Model type: ${this.model.model_type || 'custom'}`);
            console.log(`   - ${Object.keys(this.playerStats).length} players`);
            console.log(`   - ${Object.keys(this.matchupTable).length} matchups`);
            if (this.model.scenario_lookup) {
                console.log(`   - ${Object.keys(this.model.scenario_lookup).length} pre-computed scenarios`);
            }

            return true;
        } catch (error) {
            console.error('Failed to load ML predictor:', error);
            return false;
        }
    }

    /**
     * Get player stats with defaults
     */
    getPlayerStats(playerName) {
        if (this.playerStats && this.playerStats[playerName]) {
            return this.playerStats[playerName];
        }
        // Default stats for unknown players
        return {
            batting_sr: 125,
            batting_avg: 25,
            runs: 500,
            balls_faced: 400,
            bowling_economy: 8.0,
            bowling_sr: 25,
            wickets: 10,
            balls_bowled: 250
        };
    }

    /**
     * Get matchup stats
     */
    getMatchupStats(batsmanName, bowlerName) {
        if (this.matchupTable) {
            const key = `${batsmanName}|${bowlerName}`;
            if (this.matchupTable[key]) {
                return this.matchupTable[key];
            }
        }
        // Return null if no matchup history
        return null;
    }

    /**
     * Build feature vector for a delivery
     */
    buildFeatures(matchState) {
        const {
            over = 0,
            ball = 1,
            innings = 1,
            runsScored = 0,
            wicketsLost = 0,
            target = 170,
            batsmanName = 'Unknown',
            bowlerName = 'Unknown',
            venueAvgScore = 165,
            recentForm = 0
        } = matchState;

        // Get player stats
        const batStats = this.getPlayerStats(batsmanName);
        const bowlStats = this.getPlayerStats(bowlerName);

        // Get matchup stats
        const matchup = this.getMatchupStats(batsmanName, bowlerName);
        const matchupSr = matchup ? matchup.sr : batStats.batting_sr;
        const matchupBalls = matchup ? Math.min(matchup.balls / 50, 1) : 0;

        // Calculate derived features
        const phase = over < 6 ? 'POWERPLAY' : over < 15 ? 'MIDDLE' : 'DEATH';
        const ballsRemaining = (20 - over) * 6 - ball + 1;
        const ballsFaced = over * 6 + ball;
        const runRate = ballsFaced > 0 ? (runsScored / ballsFaced * 6) : 0;
        const requiredRate = innings === 2 && ballsRemaining > 0
            ? Math.max(0, (target - runsScored) / ballsRemaining * 6)
            : 0;

        // Build feature array in exact order
        const features = [
            over,
            ball,
            innings,
            phase === 'POWERPLAY' ? 1 : 0,
            phase === 'MIDDLE' ? 1 : 0,
            phase === 'DEATH' ? 1 : 0,
            runsScored,
            wicketsLost,
            runRate,
            requiredRate,
            ballsRemaining,
            batStats.batting_sr,
            batStats.batting_avg || 25,
            Math.min((batStats.balls_faced || 0) / 1000, 5),
            bowlStats.bowling_economy || 8.0,
            Math.min(bowlStats.bowling_sr || 25, 100),
            Math.min((bowlStats.balls_bowled || 0) / 500, 5),
            matchupSr,
            matchupBalls,
            venueAvgScore,
            recentForm
        ];

        return features;
    }

    /**
     * Traverse a decision tree
     */
    predictTree(features, node) {
        if (node.type === 'leaf') {
            return node.value;
        }
        if (features[node.feature] <= node.threshold) {
            return this.predictTree(features, node.left);
        } else {
            return this.predictTree(features, node.right);
        }
    }

    /**
     * Compute softmax probabilities
     */
    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        return expLogits.map(x => x / sumExp);
    }

    /**
     * Predict outcome probabilities using scenario lookup (LightGBM model)
     */
    predictProba(features) {
        if (!this.loaded || !this.model) {
            // Return baseline probabilities if model not loaded
            return [0.34, 0.37, 0.06, 0.003, 0.12, 0.06, 0.05];
        }

        // Use scenario lookup if available (new LightGBM format)
        if (this.model.scenario_lookup) {
            // Extract key features for lookup
            const over = Math.min(18, Math.floor(features[0] / 3) * 3); // Round to 0,3,6,10,15,18
            const wickets = Math.min(6, Math.floor(features[7] / 2) * 2); // Round to 0,2,4,6
            const runs = Math.min(160, Math.floor(features[6] / 40) * 40); // Round to 0,40,80,120,160

            const key = `${over}_${wickets}_${runs}`;

            if (this.model.scenario_lookup[key]) {
                let probs = [...this.model.scenario_lookup[key]];

                // Apply phase adjustments
                const currentOver = features[0];
                const phase = currentOver < 6 ? 'powerplay' : currentOver < 15 ? 'middle' : 'death';
                if (this.model.phase_adjustments && this.model.phase_adjustments[phase]) {
                    const adj = this.model.phase_adjustments[phase];
                    if (adj.DOT) probs[0] *= adj.DOT;
                    if (adj.FOUR) probs[4] *= adj.FOUR;
                    if (adj.SIX) probs[5] *= adj.SIX;
                    if (adj.WICKET) probs[6] *= adj.WICKET;
                }

                // Apply batsman/bowler adjustments based on stats
                const batSr = features[11]; // batsman_sr
                const bowlEcon = features[14]; // bowler_economy

                // Good batsman = more boundaries
                if (batSr > 140) {
                    probs[4] *= 1.1; probs[5] *= 1.15; probs[0] *= 0.9;
                } else if (batSr < 120) {
                    probs[4] *= 0.9; probs[5] *= 0.85; probs[0] *= 1.1;
                }

                // Good bowler = more dots, wickets
                if (bowlEcon < 7.5) {
                    probs[0] *= 1.1; probs[6] *= 1.1; probs[4] *= 0.9; probs[5] *= 0.9;
                } else if (bowlEcon > 9) {
                    probs[0] *= 0.9; probs[6] *= 0.9; probs[4] *= 1.1; probs[5] *= 1.1;
                }

                // Normalize
                const sum = probs.reduce((a, b) => a + b, 0);
                return probs.map(p => p / sum);
            }
        }

        // Fallback for old model format with trees
        if (this.model.trees) {
            const nClasses = this.model.n_classes;
            const F = this.model.base_probs.map(p => Math.log(p + 1e-10));

            for (const iterationTrees of this.model.trees) {
                for (let c = 0; c < nClasses; c++) {
                    const prediction = this.predictTree(features, iterationTrees[c]);
                    F[c] += this.model.learning_rate * prediction;
                }
            }
            return this.softmax(F);
        }

        // Final fallback to base probabilities
        return this.model.base_probs || [0.34, 0.37, 0.06, 0.003, 0.12, 0.06, 0.05];
    }

    /**
     * Sample an outcome based on probabilities
     */
    sampleOutcome(probs) {
        const r = Math.random();
        let cumulative = 0;

        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return this.OUTCOMES[i];
            }
        }
        return this.OUTCOMES[0]; // Fallback to DOT
    }

    /**
     * Predict next ball outcome for a match state
     * Returns: { outcome: string, probs: number[], runs: number, isWicket: boolean }
     */
    predict(matchState) {
        const features = this.buildFeatures(matchState);
        const probs = this.predictProba(features);
        const outcome = this.sampleOutcome(probs);

        // Convert outcome to runs
        const runsMap = {
            'DOT': 0,
            'SINGLE': 1,
            'DOUBLE': 2,
            'TRIPLE': 3,
            'FOUR': 4,
            'SIX': 6,
            'WICKET': 0
        };

        return {
            outcome,
            probs,
            runs: runsMap[outcome],
            isWicket: outcome === 'WICKET',
            debug: {
                features: Object.fromEntries(this.FEATURE_NAMES.map((name, i) => [name, features[i]])),
                probsNamed: Object.fromEntries(this.OUTCOMES.map((name, i) => [name, probs[i].toFixed(3)]))
            }
        };
    }

    /**
     * Calculate price change for a player based on ball outcome
     */
    calculatePriceChange(player, outcome, isOnStrike = true, isBowling = false) {
        let priceChange = 0;
        const volatility = 0.02; // Base volatility

        if (isOnStrike) {
            // Batsman price changes
            switch (outcome) {
                case 'DOT':
                    priceChange = -volatility * 0.5;
                    break;
                case 'SINGLE':
                    priceChange = volatility * 0.3;
                    break;
                case 'DOUBLE':
                    priceChange = volatility * 0.8;
                    break;
                case 'TRIPLE':
                    priceChange = volatility * 1.2;
                    break;
                case 'FOUR':
                    priceChange = volatility * 2.5;
                    break;
                case 'SIX':
                    priceChange = volatility * 3.5;
                    break;
                case 'WICKET':
                    priceChange = -volatility * 5;
                    break;
            }
        }

        if (isBowling) {
            // Bowler price changes (inverse of batting)
            switch (outcome) {
                case 'DOT':
                    priceChange = volatility * 1.0;
                    break;
                case 'SINGLE':
                    priceChange = -volatility * 0.2;
                    break;
                case 'DOUBLE':
                    priceChange = -volatility * 0.5;
                    break;
                case 'FOUR':
                    priceChange = -volatility * 1.5;
                    break;
                case 'SIX':
                    priceChange = -volatility * 2.5;
                    break;
                case 'WICKET':
                    priceChange = volatility * 4;
                    break;
            }
        }

        // Add some random noise for GBM-like behavior
        const noise = (Math.random() - 0.5) * volatility * 0.5;

        return priceChange + noise;
    }
}

// Export for use in game
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CricketMLPredictor;
}
