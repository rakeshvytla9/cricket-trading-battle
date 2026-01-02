#!/usr/bin/env python3
"""
Create training data for Cricket ML models
Engineers 15+ features from raw ball-by-ball data
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Paths
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ML_DIR = DATA_DIR.parent / "ml"

# Outcome mapping
OUTCOME_CLASSES = ['DOT', 'SINGLE', 'DOUBLE', 'TRIPLE', 'FOUR', 'SIX', 'WICKET']

def load_player_career_stats() -> Dict[str, Dict]:
    """Load pre-computed player career statistics"""
    players = {}
    with open(PROCESSED_DIR / 'players.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            players[row['player_name']] = {
                'batting_sr': float(row['strike_rate']) if row['strike_rate'] else 0,
                'batting_avg': float(row['average']) if row['average'] else 0,
                'runs': int(row['runs_scored']) if row['runs_scored'] else 0,
                'balls_faced': int(row['balls_faced']) if row['balls_faced'] else 0,
                'bowling_economy': float(row['economy']) if row['economy'] else 0,
                'bowling_sr': (float(row['balls_bowled']) / float(row['wickets'])) if row['wickets'] and float(row['wickets']) > 0 else 100,
                'wickets': int(row['wickets']) if row['wickets'] else 0,
                'balls_bowled': int(row['balls_bowled']) if row['balls_bowled'] else 0
            }
    return players

def load_venue_stats() -> Dict[str, Dict]:
    """Load venue average scores (we'll compute from data)"""
    venues = {}
    venue_runs = defaultdict(list)
    
    with open(PROCESSED_DIR / 'deliveries.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        match_innings_runs = defaultdict(int)
        
        for row in reader:
            key = (row['match_id'], row['innings'], row['venue'])
            match_innings_runs[key] += int(row['runs_total'])
        
        for (match_id, innings, venue), runs in match_innings_runs.items():
            venue_runs[venue].append(runs)
    
    for venue, runs_list in venue_runs.items():
        venues[venue] = {
            'avg_innings_score': sum(runs_list) / len(runs_list) if runs_list else 160,
            'matches': len(runs_list) // 2
        }
    
    return venues

def compute_matchup_stats(deliveries: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """Compute batsman vs bowler historical matchups"""
    matchups = defaultdict(lambda: {'balls': 0, 'runs': 0, 'wickets': 0})
    
    for d in deliveries:
        key = (d['batsman'], d['bowler'])
        matchups[key]['balls'] += 1
        matchups[key]['runs'] += d['runs_batter']
        if d['wicket_type'] and d['player_dismissed'] == d['batsman']:
            matchups[key]['wickets'] += 1
    
    # Compute SR for each matchup
    result = {}
    for key, stats in matchups.items():
        if stats['balls'] >= 6:  # At least 1 over faced
            result[key] = {
                'balls': stats['balls'],
                'runs': stats['runs'],
                'wickets': stats['wickets'],
                'sr': (stats['runs'] / stats['balls'] * 100) if stats['balls'] > 0 else 0
            }
    
    return result

def get_outcome_class(runs_batter: int, wicket_type: str) -> str:
    """Convert ball result to outcome class"""
    if wicket_type:
        return 'WICKET'
    elif runs_batter == 0:
        return 'DOT'
    elif runs_batter == 1:
        return 'SINGLE'
    elif runs_batter == 2:
        return 'DOUBLE'
    elif runs_batter == 3:
        return 'TRIPLE'
    elif runs_batter == 4:
        return 'FOUR'
    elif runs_batter == 6:
        return 'SIX'
    else:
        return 'DOT'  # Rare cases

def get_match_phase(over: int) -> str:
    """Get match phase from over number"""
    if over < 6:
        return 'POWERPLAY'
    elif over < 15:
        return 'MIDDLE'
    else:
        return 'DEATH'

def engineer_features(deliveries: List[Dict], player_stats: Dict, venue_stats: Dict, matchup_stats: Dict) -> List[Dict]:
    """Engineer all features for each delivery"""
    
    # Group deliveries by match and innings for running totals
    match_innings = defaultdict(list)
    for d in deliveries:
        key = (d['match_id'], d['innings'])
        match_innings[key].append(d)
    
    features_list = []
    
    for (match_id, innings), balls in match_innings.items():
        # Sort by over and ball
        balls = sorted(balls, key=lambda x: (x['over'], x['ball']))
        
        running_runs = 0
        running_wickets = 0
        recent_runs = []  # Last 5 balls
        
        for i, d in enumerate(balls):
            over = d['over']
            ball_in_over = d['ball']
            
            # Get player stats
            bat_stats = player_stats.get(d['batsman'], {})
            bowl_stats = player_stats.get(d['bowler'], {})
            
            # Get venue stats
            v_stats = venue_stats.get(d['venue'], {'avg_innings_score': 160})
            
            # Get matchup stats
            matchup_key = (d['batsman'], d['bowler'])
            m_stats = matchup_stats.get(matchup_key, {'sr': bat_stats.get('batting_sr', 125)})
            
            # Calculate required rate (for 2nd innings)
            target = 170  # Approximate, we don't have actual targets
            balls_remaining = (20 - over) * 6 - ball_in_over + 1
            required_rate = ((target - running_runs) / balls_remaining * 6) if balls_remaining > 0 and innings == 2 else 0
            
            # Current run rate
            balls_faced_so_far = over * 6 + ball_in_over
            current_rr = (running_runs / balls_faced_so_far * 6) if balls_faced_so_far > 0 else 0
            
            # Recent form (runs in last 5 balls by this batsman)
            recent_form = sum(recent_runs[-5:]) if recent_runs else 0
            
            # Phase encoding
            phase = get_match_phase(over)
            phase_powerplay = 1 if phase == 'POWERPLAY' else 0
            phase_middle = 1 if phase == 'MIDDLE' else 0
            phase_death = 1 if phase == 'DEATH' else 0
            
            # Target outcome
            outcome = get_outcome_class(d['runs_batter'], d['wicket_type'])
            outcome_idx = OUTCOME_CLASSES.index(outcome)
            
            features = {
                # Identifiers (not for training)
                'match_id': match_id,
                'delivery_id': f"{match_id}_{innings}_{over}_{ball_in_over}",
                
                # Core features
                'over': over,
                'ball_in_over': ball_in_over,
                'innings': innings,
                'phase_powerplay': phase_powerplay,
                'phase_middle': phase_middle,
                'phase_death': phase_death,
                
                # Match state
                'runs_scored': running_runs,
                'wickets_lost': running_wickets,
                'run_rate': round(current_rr, 2),
                'required_rate': round(max(0, required_rate), 2),
                'balls_remaining': balls_remaining,
                
                # Batsman features
                'batsman_sr': bat_stats.get('batting_sr', 125),
                'batsman_avg': bat_stats.get('batting_avg', 25),
                'batsman_experience': min(bat_stats.get('balls_faced', 0) / 1000, 5),  # Normalized
                
                # Bowler features
                'bowler_economy': bowl_stats.get('bowling_economy', 8.0),
                'bowler_sr': min(bowl_stats.get('bowling_sr', 25), 100),  # Cap at 100
                'bowler_experience': min(bowl_stats.get('balls_bowled', 0) / 500, 5),  # Normalized
                
                # Matchup features
                'matchup_sr': m_stats.get('sr', bat_stats.get('batting_sr', 125)),
                'matchup_balls': min(m_stats.get('balls', 0) / 50, 1),  # Normalized
                
                # Venue features
                'venue_avg_score': v_stats.get('avg_innings_score', 160),
                
                # Recent form
                'recent_form': recent_form,
                
                # Target
                'outcome': outcome,
                'outcome_idx': outcome_idx
            }
            
            features_list.append(features)
            
            # Update running totals
            running_runs += d['runs_total']
            if d['wicket_type']:
                running_wickets += 1
            recent_runs.append(d['runs_batter'])
    
    return features_list

def main():
    print("=" * 60)
    print("Creating Training Data for Cricket ML Model")
    print("=" * 60)
    
    # Load raw deliveries
    print("\n1. Loading deliveries...")
    deliveries = []
    with open(PROCESSED_DIR / 'deliveries.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            deliveries.append({
                'match_id': row['match_id'],
                'date': row['date'],
                'season': row['season'],
                'venue': row['venue'],
                'innings': int(row['innings']),
                'over': int(row['over']),
                'ball': int(row['ball']),
                'batting_team': row['batting_team'],
                'bowling_team': row['bowling_team'],
                'batsman': row['batsman'],
                'bowler': row['bowler'],
                'runs_batter': int(row['runs_batter']),
                'runs_extras': int(row['runs_extras']),
                'runs_total': int(row['runs_total']),
                'wicket_type': row['wicket_type'],
                'player_dismissed': row['player_dismissed']
            })
    print(f"   Loaded {len(deliveries)} deliveries")
    
    # Load player stats
    print("\n2. Loading player statistics...")
    player_stats = load_player_career_stats()
    print(f"   Loaded stats for {len(player_stats)} players")
    
    # Load venue stats
    print("\n3. Computing venue statistics...")
    venue_stats = load_venue_stats()
    print(f"   Computed stats for {len(venue_stats)} venues")
    
    # Compute matchup stats
    print("\n4. Computing bowler-batsman matchups...")
    matchup_stats = compute_matchup_stats(deliveries)
    print(f"   Computed {len(matchup_stats)} matchups with 6+ balls")
    
    # Engineer features
    print("\n5. Engineering features...")
    features = engineer_features(deliveries, player_stats, venue_stats, matchup_stats)
    print(f"   Created {len(features)} training samples")
    
    # Split data (by match to avoid leakage)
    print("\n6. Splitting train/val/test...")
    matches = list(set(f['match_id'] for f in features))
    matches.sort()
    
    n_train = int(len(matches) * 0.7)
    n_val = int(len(matches) * 0.15)
    
    train_matches = set(matches[:n_train])
    val_matches = set(matches[n_train:n_train + n_val])
    test_matches = set(matches[n_train + n_val:])
    
    train_data = [f for f in features if f['match_id'] in train_matches]
    val_data = [f for f in features if f['match_id'] in val_matches]
    test_data = [f for f in features if f['match_id'] in test_matches]
    
    print(f"   Train: {len(train_data)} samples ({len(train_matches)} matches)")
    print(f"   Val:   {len(val_data)} samples ({len(val_matches)} matches)")
    print(f"   Test:  {len(test_data)} samples ({len(test_matches)} matches)")
    
    # Save to CSV
    print("\n7. Saving training data...")
    ML_DIR.mkdir(parents=True, exist_ok=True)
    
    feature_cols = [
        'over', 'ball_in_over', 'innings', 'phase_powerplay', 'phase_middle', 'phase_death',
        'runs_scored', 'wickets_lost', 'run_rate', 'required_rate', 'balls_remaining',
        'batsman_sr', 'batsman_avg', 'batsman_experience',
        'bowler_economy', 'bowler_sr', 'bowler_experience',
        'matchup_sr', 'matchup_balls', 'venue_avg_score', 'recent_form',
        'outcome', 'outcome_idx'
    ]
    
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        filepath = ML_DIR / f'{name}_data.csv'
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=feature_cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        print(f"   Saved {filepath.name}")
    
    # Save player stats and matchups as JSON for game integration
    print("\n8. Exporting player stats and matchups...")
    
    # Top 50 players for the game
    top_players = sorted(player_stats.items(), key=lambda x: x[1]['runs'] + x[1]['wickets'] * 20, reverse=True)[:50]
    player_stats_export = {name: stats for name, stats in top_players}
    
    with open(DATA_DIR / 'player_stats.json', 'w', encoding='utf-8') as f:
        json.dump(player_stats_export, f, indent=2)
    print(f"   Saved player_stats.json ({len(player_stats_export)} players)")
    
    # Top matchups (at least 10 balls)
    top_matchups = {f"{k[0]}|{k[1]}": v for k, v in matchup_stats.items() if v['balls'] >= 10}
    with open(DATA_DIR / 'matchup_table.json', 'w', encoding='utf-8') as f:
        json.dump(top_matchups, f, indent=2)
    print(f"   Saved matchup_table.json ({len(top_matchups)} matchups)")
    
    print("\n" + "=" * 60)
    print("✅ Training data created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
