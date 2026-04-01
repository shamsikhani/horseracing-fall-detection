"""
Generate a curated list of 100 videos for the dataset.
Exports to Excel with all necessary metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Read grief data
file_path = Path(r"c:\Users\shams\Documents\Horseracing\RACE_GRIEFS.xlsx")
df = pd.read_excel(file_path, engine='openpyxl')

# Create unique race identifier
df['race_id'] = df['Venue'] + '_' + df['Date'].astype(str) + '_' + df['Time'].astype(str)

# Get one row per race with aggregated grief info
race_summary = df.groupby('race_id').agg({
    'Venue': 'first',
    'Date': 'first',
    'Time': 'first',
    'Race Name': 'first',
    'Going': 'first',
    'Furlongs': 'first',
    'Grief': lambda x: ', '.join(x.astype(str)),
    'Horse': lambda x: ', '.join(x.astype(str))
}).reset_index()

# Add fall indicator
race_summary['has_fall'] = race_summary['Grief'].str.lower().str.contains('fell', na=False)

# Count falls per race
race_summary['num_falls'] = race_summary['Grief'].str.lower().str.count('fell')

# Count total events per race
race_summary['num_events'] = race_summary['Grief'].str.count(',') + 1

# Current 12 videos (already have these)
current_videos = [
    ("Naas", "2024-01-12", "13:12:00"),
    ("Naas", "2024-01-12", "15:12:00"),
    ("Punchestown", "2024-01-15", "14:55:00"),
    ("Navan", "2024-01-20", "14:30:00"),
    ("Fairyhouse", "2024-01-27", "13:05:00"),
    ("Limerick", "2024-01-30", "14:00:00"),
    ("Fairyhouse", "2024-01-01", "12:40:00"),
    ("Clonmel", "2024-01-11", "15:20:00"),
    ("Punchestown", "2024-01-15", "13:25:00"),
    ("Navan", "2024-01-20", "15:42:00"),
    ("Down Royal", "2024-01-23", "15:30:00"),
    ("Gowran Park", "2024-01-25", "13:40:00"),
]

# Mark current videos
race_summary['already_have'] = False
for venue, date, time in current_videos:
    mask = (race_summary['Venue'] == venue) & \
           (race_summary['Date'].astype(str) == date) & \
           (race_summary['Time'].astype(str) == time)
    race_summary.loc[mask, 'already_have'] = True

# Separate into fall and no-fall
fall_races = race_summary[race_summary['has_fall'] == True].copy()
no_fall_races = race_summary[race_summary['has_fall'] == False].copy()

# Count what we already have
current_fall = race_summary[race_summary['already_have'] & race_summary['has_fall']].shape[0]
current_no_fall = race_summary[race_summary['already_have'] & ~race_summary['has_fall']].shape[0]

print(f"Current inventory:")
print(f"  Fall videos: {current_fall}")
print(f"  No-fall videos: {current_no_fall}")
print(f"  Total: {current_fall + current_no_fall}")

# Target: 35 fall, 65 no-fall
target_fall = 35
target_no_fall = 65

need_fall = target_fall - current_fall
need_no_fall = target_no_fall - current_no_fall

print(f"\nTarget for 100 videos:")
print(f"  Need {need_fall} more fall videos")
print(f"  Need {need_no_fall} more no-fall videos")

# Sample new videos (stratified by venue for diversity)
np.random.seed(42)  # Reproducible selection

# Sample fall videos
available_fall = fall_races[~fall_races['already_have']]
if len(available_fall) >= need_fall:
    # Stratify by venue
    fall_venues = available_fall['Venue'].value_counts()
    selected_fall = []
    
    # Calculate proportional allocation
    for venue in fall_venues.index:
        venue_races = available_fall[available_fall['Venue'] == venue]
        n_from_venue = max(1, int(need_fall * len(venue_races) / len(available_fall)))
        n_from_venue = min(n_from_venue, len(venue_races))
        
        sampled = venue_races.sample(n=n_from_venue, random_state=42)
        selected_fall.append(sampled)
    
    selected_fall_df = pd.concat(selected_fall)
    
    # If we have too many, randomly drop some
    if len(selected_fall_df) > need_fall:
        selected_fall_df = selected_fall_df.sample(n=need_fall, random_state=42)
    # If we have too few, add more randomly
    elif len(selected_fall_df) < need_fall:
        remaining = available_fall[~available_fall.index.isin(selected_fall_df.index)]
        extra = remaining.sample(n=need_fall - len(selected_fall_df), random_state=42)
        selected_fall_df = pd.concat([selected_fall_df, extra])
else:
    print(f"WARNING: Only {len(available_fall)} fall videos available, need {need_fall}")
    selected_fall_df = available_fall

# Sample no-fall videos (stratified by grief type)
available_no_fall = no_fall_races[~no_fall_races['already_have']]

# Categorize no-fall videos by primary grief type
def get_primary_grief(grief_str):
    grief_lower = str(grief_str).lower()
    if 'pulled up' in grief_lower:
        return 'pulled_up'
    elif 'unseated' in grief_lower:
        return 'unseated_rider'
    elif 'brought down' in grief_lower:
        return 'brought_down'
    else:
        return 'other'

available_no_fall['primary_grief'] = available_no_fall['Grief'].apply(get_primary_grief)

# Target distribution for no-fall videos
# 40 pulled-up, 15 unseated, 10 other
grief_targets = {
    'pulled_up': min(40 - current_no_fall, need_no_fall * 0.6),
    'unseated_rider': min(15, need_no_fall * 0.25),
    'other': min(10, need_no_fall * 0.15)
}

selected_no_fall = []
for grief_type, target_n in grief_targets.items():
    target_n = int(target_n)
    available = available_no_fall[available_no_fall['primary_grief'] == grief_type]
    
    if len(available) >= target_n:
        sampled = available.sample(n=target_n, random_state=42)
    else:
        sampled = available
    
    selected_no_fall.append(sampled)

selected_no_fall_df = pd.concat(selected_no_fall)

# If we still need more, fill with any available
if len(selected_no_fall_df) < need_no_fall:
    remaining = available_no_fall[~available_no_fall.index.isin(selected_no_fall_df.index)]
    extra_needed = need_no_fall - len(selected_no_fall_df)
    if len(remaining) >= extra_needed:
        extra = remaining.sample(n=extra_needed, random_state=42)
        selected_no_fall_df = pd.concat([selected_no_fall_df, extra])
    else:
        selected_no_fall_df = pd.concat([selected_no_fall_df, remaining])

# Combine all selected videos
current_videos_df = race_summary[race_summary['already_have']].copy()
current_videos_df['status'] = 'ALREADY_HAVE'

selected_fall_df['status'] = 'NEED_TO_OBTAIN'
selected_no_fall_df['status'] = 'NEED_TO_OBTAIN'

final_selection = pd.concat([current_videos_df, selected_fall_df, selected_no_fall_df])

# Create video ID in standard format
def create_video_id(row):
    date_str = pd.to_datetime(row['Date']).strftime('%Y%m%d')
    time_str = pd.to_datetime(row['Time'], format='%H:%M:%S').strftime('%H%M')
    
    # Map venue to 3-letter code
    venue_codes = {
        'Naas': 'naa', 'Punchestown': 'pun', 'Navan': 'nav',
        'Fairyhouse': 'fai', 'Limerick': 'lim', 'Clonmel': 'clo',
        'Down Royal': 'dnr', 'Gowran Park': 'gow', 'Leopardstown': 'leo',
        'Thurles': 'thu', 'Tramore': 'tra', 'Kilbeggan': 'kil',
        'Cork': 'cor', 'Galway': 'gal', 'Listowel': 'lis',
        'Sligo': 'sli', 'Ballinrobe': 'bal', 'Killarney': 'kla',
        'Roscommon': 'ros', 'Tipperary': 'tip', 'Wexford': 'wex',
        'Bellewstown': 'bel', 'Downpatrick': 'dwn', 'Laytown': 'lay'
    }
    
    venue_code = venue_codes.get(row['Venue'], row['Venue'][:3].lower())
    
    return f"{date_str}_{venue_code}_{time_str}_txop_3F"

final_selection['video_id'] = final_selection.apply(create_video_id, axis=1)

# Add class label
final_selection['class'] = final_selection['has_fall'].apply(lambda x: 'FALL' if x else 'NO_FALL')

# Sort by status (already have first), then by class, then by date
final_selection = final_selection.sort_values(['status', 'class', 'Date'])

# Create output dataframe with clean columns
output_df = final_selection[[
    'video_id', 'status', 'class', 'Venue', 'Date', 'Time',
    'Race Name', 'num_falls', 'num_events', 'Grief', 'Going', 'Furlongs'
]].copy()

output_df.columns = [
    'Video_ID', 'Status', 'Class', 'Venue', 'Date', 'Time',
    'Race_Name', 'Num_Falls', 'Num_Events', 'All_Grief_Events', 'Going', 'Distance_Furlongs'
]

# Save to Excel
output_path = Path(r"c:\Users\shams\Documents\Horseracing\video_selection_100.xlsx")
output_df.to_excel(output_path, index=False, sheet_name='Video Selection')

# Print summary
print(f"\n{'='*80}")
print("FINAL SELECTION SUMMARY")
print(f"{'='*80}")
print(f"\nTotal videos selected: {len(output_df)}")
print(f"\nBy status:")
print(output_df['Status'].value_counts())
print(f"\nBy class:")
print(output_df['Class'].value_counts())
print(f"\nBy venue (top 10):")
print(output_df['Venue'].value_counts().head(10))

print(f"\n{'='*80}")
print(f"Excel file saved to: {output_path}")
print(f"{'='*80}")

# Create a summary sheet
summary_data = {
    'Metric': [
        'Total Videos',
        'Already Have',
        'Need to Obtain',
        '',
        'Fall Videos',
        'No-Fall Videos',
        '',
        'Fall Prevalence',
        'Avg Falls per Fall Video',
        'Avg Events per Video',
        '',
        'Unique Venues'
    ],
    'Value': [
        len(output_df),
        (output_df['Status'] == 'ALREADY_HAVE').sum(),
        (output_df['Status'] == 'NEED_TO_OBTAIN').sum(),
        '',
        (output_df['Class'] == 'FALL').sum(),
        (output_df['Class'] == 'NO_FALL').sum(),
        '',
        f"{(output_df['Class'] == 'FALL').sum() / len(output_df) * 100:.1f}%",
        f"{output_df[output_df['Class'] == 'FALL']['Num_Falls'].mean():.2f}",
        f"{output_df['Num_Events'].mean():.2f}",
        '',
        output_df['Venue'].nunique()
    ]
}

summary_df = pd.DataFrame(summary_data)

# Write both sheets
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    output_df.to_excel(writer, sheet_name='Video Selection', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"\nExcel file created with 2 sheets:")
print(f"  1. 'Video Selection' - Full list of 100 videos")
print(f"  2. 'Summary' - Dataset statistics")
