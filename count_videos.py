import pandas as pd

df = pd.read_excel('video_selection_100.xlsx')

FALL_INCIDENTS = {'fell', 'brought down', 'unseated rider', 'slipped up'}

fall_count = 0
no_fall_count = 0
fall_videos = []
no_fall_videos = []

for _, row in df.iterrows():
    video_id = row['Video_ID']
    events_str = str(row['All_Grief_Events'])
    events_list = [e.strip() for e in events_str.split(',')]
    has_fall = any(event in FALL_INCIDENTS for event in events_list)
    
    if has_fall:
        fall_count += 1
        fall_videos.append((video_id, events_str))
    else:
        no_fall_count += 1
        no_fall_videos.append((video_id, events_str))

print(f'TOTAL VIDEOS: {len(df)}')
print(f'\nFALL VIDEOS (fell/brought down/unseated rider/slipped up): {fall_count}')
print(f'NO-FALL VIDEOS (pulled up/refused/carried out/ran out): {no_fall_count}')
print(f'\nBreakdown by original folder:')
print(f'  Fell folder: {sum(df["Class"] == "FALL")}')
print(f'  Pulled-up folder: {sum(df["Class"] == "NO_FALL")}')

print(f'\n=== RECLASSIFICATION DETAILS ===')
print(f'\nVideos moved from Pulled-up folder to FALL category:')
pulled_up_with_falls = [(vid, events) for vid, events in fall_videos if vid in df[df['Class'] == 'NO_FALL']['Video_ID'].values]
print(f'Count: {len(pulled_up_with_falls)}')
for vid, events in pulled_up_with_falls[:10]:
    print(f'  {vid}: {events}')
if len(pulled_up_with_falls) > 10:
    print(f'  ... and {len(pulled_up_with_falls) - 10} more')
