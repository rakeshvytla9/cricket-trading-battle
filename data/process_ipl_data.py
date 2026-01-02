#!/usr/bin/env python3
"""
Process IPL ball-by-ball data from Cricsheet JSON files
Filters for seasons 2020-2024 and exports to CSV
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Filter for these seasons
TARGET_SEASONS = ["2020", "2021", "2022", "2023", "2024"]

def parse_match_file(filepath: Path) -> dict:
    """Parse a single Cricsheet JSON match file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_deliveries(match_data: dict, match_id: str) -> list:
    """Extract all deliveries from a match"""
    deliveries = []
    
    info = match_data.get('info', {})
    dates = info.get('dates', [])
    match_date = dates[0] if dates else ''
    venue = info.get('venue', '')
    city = info.get('city', '')
    teams = info.get('teams', [])
    season = info.get('season', '')
    
    # Check if in target seasons
    season_str = str(season)
    if season_str not in TARGET_SEASONS:
        return []
    
    innings_data = match_data.get('innings', [])
    
    for innings_num, innings in enumerate(innings_data, 1):
        batting_team = innings.get('team', '')
        bowling_team = [t for t in teams if t != batting_team]
        bowling_team = bowling_team[0] if bowling_team else ''
        
        overs = innings.get('overs', [])
        for over_data in overs:
            over_num = over_data.get('over', 0)
            
            for ball_num, delivery in enumerate(over_data.get('deliveries', []), 1):
                batter = delivery.get('batter', '')
                bowler = delivery.get('bowler', '')
                non_striker = delivery.get('non_striker', '')
                runs = delivery.get('runs', {})
                
                runs_batter = runs.get('batter', 0)
                runs_extras = runs.get('extras', 0)
                runs_total = runs.get('total', 0)
                
                # Extras detail
                extras = delivery.get('extras', {})
                extras_type = ','.join(extras.keys()) if extras else ''
                
                # Wicket info
                wickets = delivery.get('wickets', [])
                wicket_type = ''
                player_dismissed = ''
                if wickets:
                    wicket_type = wickets[0].get('kind', '')
                    player_dismissed = wickets[0].get('player_out', '')
                
                deliveries.append({
                    'match_id': match_id,
                    'date': match_date,
                    'season': season_str,
                    'venue': venue,
                    'city': city,
                    'innings': innings_num,
                    'over': over_num,
                    'ball': ball_num,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'batsman': batter,
                    'non_striker': non_striker,
                    'bowler': bowler,
                    'runs_batter': runs_batter,
                    'runs_extras': runs_extras,
                    'runs_total': runs_total,
                    'extras_type': extras_type,
                    'wicket_type': wicket_type,
                    'player_dismissed': player_dismissed
                })
    
    return deliveries

def calculate_player_stats(all_deliveries: list) -> list:
    """Calculate aggregated player statistics"""
    batting_stats = defaultdict(lambda: {
        'matches': set(),
        'innings': set(),
        'runs': 0,
        'balls': 0,
        'fours': 0,
        'sixes': 0,
        'dismissals': 0
    })
    
    bowling_stats = defaultdict(lambda: {
        'matches': set(),
        'overs': set(),
        'runs': 0,
        'wickets': 0,
        'balls': 0
    })
    
    for d in all_deliveries:
        match_id = d['match_id']
        innings_key = f"{match_id}_{d['innings']}"
        over_key = f"{match_id}_{d['innings']}_{d['over']}"
        
        # Batting stats
        batsman = d['batsman']
        batting_stats[batsman]['matches'].add(match_id)
        batting_stats[batsman]['innings'].add(innings_key)
        batting_stats[batsman]['runs'] += d['runs_batter']
        batting_stats[batsman]['balls'] += 1
        
        if d['runs_batter'] == 4:
            batting_stats[batsman]['fours'] += 1
        elif d['runs_batter'] == 6:
            batting_stats[batsman]['sixes'] += 1
        
        if d['player_dismissed'] == batsman:
            batting_stats[batsman]['dismissals'] += 1
        
        # Bowling stats
        bowler = d['bowler']
        bowling_stats[bowler]['matches'].add(match_id)
        bowling_stats[bowler]['overs'].add(over_key)
        bowling_stats[bowler]['runs'] += d['runs_total']
        bowling_stats[bowler]['balls'] += 1
        
        if d['wicket_type'] and d['wicket_type'] not in ['run out', 'retired hurt', 'retired out']:
            bowling_stats[bowler]['wickets'] += 1
    
    # Combine into player records
    all_players = set(batting_stats.keys()) | set(bowling_stats.keys())
    players = []
    
    for player in all_players:
        bat = batting_stats.get(player, {})
        bowl = bowling_stats.get(player, {})
        
        bat_matches = len(bat.get('matches', set()))
        bat_innings = len(bat.get('innings', set()))
        runs = bat.get('runs', 0)
        balls_faced = bat.get('balls', 0)
        fours = bat.get('fours', 0)
        sixes = bat.get('sixes', 0)
        dismissals = bat.get('dismissals', 0)
        
        strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0
        average = (runs / dismissals) if dismissals > 0 else runs
        
        bowl_matches = len(bowl.get('matches', set()))
        overs_bowled = len(bowl.get('overs', set()))
        runs_conceded = bowl.get('runs', 0)
        wickets = bowl.get('wickets', 0)
        balls_bowled = bowl.get('balls', 0)
        
        economy = (runs_conceded / (balls_bowled / 6)) if balls_bowled > 0 else 0
        bowl_average = (runs_conceded / wickets) if wickets > 0 else 0
        
        players.append({
            'player_name': player,
            'matches': max(bat_matches, bowl_matches),
            'innings_batted': bat_innings,
            'runs_scored': runs,
            'balls_faced': balls_faced,
            'strike_rate': round(strike_rate, 2),
            'average': round(average, 2),
            'fours': fours,
            'sixes': sixes,
            'overs_bowled': overs_bowled,
            'balls_bowled': balls_bowled,
            'runs_conceded': runs_conceded,
            'wickets': wickets,
            'economy': round(economy, 2),
            'bowling_average': round(bowl_average, 2)
        })
    
    return sorted(players, key=lambda x: x['runs_scored'], reverse=True)

def extract_venues(all_deliveries: list) -> list:
    """Extract unique venue information"""
    venues = {}
    for d in all_deliveries:
        venue = d['venue']
        if venue and venue not in venues:
            venues[venue] = {
                'venue': venue,
                'city': d['city'],
                'matches': 0
            }
        if venue:
            venues[venue]['matches'] += 1
    
    # Convert to list with match count
    venue_list = []
    venue_matches = defaultdict(set)
    for d in all_deliveries:
        if d['venue']:
            venue_matches[d['venue']].add(d['match_id'])
    
    for venue, data in venues.items():
        venue_list.append({
            'venue': venue,
            'city': data['city'],
            'matches': len(venue_matches[venue])
        })
    
    return sorted(venue_list, key=lambda x: x['matches'], reverse=True)

def write_csv(data: list, filepath: Path, fieldnames: list):
    """Write data to CSV file"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Written {len(data)} rows to {filepath.name}")

def main():
    print("=" * 50)
    print("IPL Data Processor")
    print(f"Processing seasons: {', '.join(TARGET_SEASONS)}")
    print("=" * 50)
    
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(RAW_DIR.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {RAW_DIR}")
        print("Run download_ipl_data.py first!")
        sys.exit(1)
    
    print(f"Found {len(json_files)} match files")
    
    # Process all matches
    all_deliveries = []
    matches_processed = 0
    
    for filepath in json_files:
        match_id = filepath.stem
        try:
            match_data = parse_match_file(filepath)
            deliveries = extract_deliveries(match_data, match_id)
            if deliveries:
                all_deliveries.extend(deliveries)
                matches_processed += 1
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
    
    print(f"\n📊 Processed {matches_processed} matches from target seasons")
    print(f"📊 Total deliveries: {len(all_deliveries)}")
    
    if not all_deliveries:
        print("❌ No deliveries found for target seasons")
        sys.exit(1)
    
    # Write deliveries CSV
    delivery_fields = [
        'match_id', 'date', 'season', 'venue', 'city', 'innings', 'over', 'ball',
        'batting_team', 'bowling_team', 'batsman', 'non_striker', 'bowler',
        'runs_batter', 'runs_extras', 'runs_total', 'extras_type',
        'wicket_type', 'player_dismissed'
    ]
    write_csv(all_deliveries, PROCESSED_DIR / 'deliveries.csv', delivery_fields)
    
    # Calculate and write player stats
    print("\nCalculating player statistics...")
    player_stats = calculate_player_stats(all_deliveries)
    player_fields = [
        'player_name', 'matches', 'innings_batted', 'runs_scored', 'balls_faced',
        'strike_rate', 'average', 'fours', 'sixes',
        'overs_bowled', 'balls_bowled', 'runs_conceded', 'wickets', 'economy', 'bowling_average'
    ]
    write_csv(player_stats, PROCESSED_DIR / 'players.csv', player_fields)
    
    # Extract and write venues
    venues = extract_venues(all_deliveries)
    venue_fields = ['venue', 'city', 'matches']
    write_csv(venues, PROCESSED_DIR / 'venues.csv', venue_fields)
    
    print(f"\n✅ All data exported to {PROCESSED_DIR}/")
    print("\nTop 10 run scorers (2020-2024):")
    for i, p in enumerate(player_stats[:10], 1):
        print(f"  {i}. {p['player_name']}: {p['runs_scored']} runs @ SR {p['strike_rate']}")

if __name__ == "__main__":
    main()
