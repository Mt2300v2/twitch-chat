import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# Define a set of common Twitch emotes (all lower-case for case-insensitive matching)
COMMON_EMOTES = {"kappa", "pogchamp", "lul", "residentsleeper", "biblethump",
                 "4head", "kkona", "kreygasm", "trihard", "seemsgood", "keepo"}

def compute_message_metrics(msg, common_emotes):
    """
    Given a message string, compute and return:
      - emote_density: fraction of words that are in the common_emotes set.
      - repetition_ratio: fraction of words that appear more than once.
      - unique_ratio: fraction of unique words (lowercase) among all words.
    If the message is empty or not a string, returns (0, 0, 0).
    """
    if not isinstance(msg, str):
        return 0, 0, 0
    words = msg.split()
    total = len(words)
    if total == 0:
        return 0, 0, 0
    emote_count = sum(1 for w in words if w.lower() in common_emotes)
    emote_density = emote_count / total
    counts = Counter(w.lower() for w in words)
    repeated = sum(count for count in counts.values() if count > 1)
    repetition_ratio = repeated / total
    unique_ratio = len(set(w.lower() for w in words)) / total
    return emote_density, repetition_ratio, unique_ratio

def process_csv_files(file_paths, chunksize=10**6):
    """
    Process multiple CSV files (with columns: timestamp, channel, username, message)
    in chunks and aggregate data for plotting.
    """
    # Aggregation dictionaries and accumulators:
    time_counts = {}          # messages per minute
    channel_counts = {}       # total messages per channel
    user_counts = {}          # total messages per user
    hour_counts = {}          # messages by hour-of-day
    day_of_week_counts = {}   # messages by day-of-week
    hour_day_counts = {}      # (day_of_week, hour) counts for heatmap

    # Histogram bins and accumulators:
    ml_bins = list(range(0, 310, 10)) + [np.inf]  # Message length bins
    ml_hist = np.zeros(len(ml_bins)-1, dtype=int)
    wc_bins = [0, 2, 4, 6, 8, 10, 12, np.inf]      # Word count bins
    wc_hist = np.zeros(len(wc_bins)-1, dtype=int)

    # Scatter plot samples:
    scatter_lengths = []        # message lengths
    scatter_word_counts = []    # word counts per message
    scatter_unique_ratios = []  # unique word ratio per message

    # For box plots: reservoir sample of message lengths per channel.
    channel_box_samples = {}
    sample_per_channel_chunk = 100
    max_samples_per_channel = 1000

    # Word frequency counter for word cloud and top-word graphs.
    word_freq = Counter()

    # New aggregations for extra graphs:
    all_emote_densities = []         # all messages' emote densities
    all_repetition_ratios = []         # all messages' repetition ratios
    all_unique_ratios = []             # all messages' unique ratios
    channel_emote_densities = {}       # channel -> list of emote densities
    channel_repetition_ratios = {}     # channel -> list of repetition ratios
    channel_unique_ratios = {}         # channel -> list of unique ratios

    for file_path in file_paths:
        print(f"Processing {file_path}...")
        for chunk in pd.read_csv(file_path, header=None,
                                 names=['timestamp', 'channel', 'username', 'message'],
                                 chunksize=chunksize):
            # Convert timestamp to datetime; drop rows with invalid timestamps.
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
            chunk = chunk.dropna(subset=['timestamp'])
            
            # Use 'min' instead of 'T' to floor timestamps.
            chunk['minute'] = chunk['timestamp'].dt.floor('min')
            for minute, count in chunk.groupby('minute').size().items():
                time_counts[minute] = time_counts.get(minute, 0) + count

            # Count messages per channel and per user.
            for channel, count in chunk['channel'].value_counts().items():
                channel_counts[channel] = channel_counts.get(channel, 0) + count
            for user, count in chunk['username'].value_counts().items():
                user_counts[user] = user_counts.get(user, 0) + count

            # Count messages by hour of day.
            chunk['hour'] = chunk['timestamp'].dt.hour
            for hour, count in chunk['hour'].value_counts().items():
                hour_counts[hour] = hour_counts.get(hour, 0) + count

            # Count messages by day of week.
            chunk['day_of_week'] = chunk['timestamp'].dt.day_name()
            for day, count in chunk['day_of_week'].value_counts().items():
                day_of_week_counts[day] = day_of_week_counts.get(day, 0) + count

            # Build (day, hour) counts for heatmap.
            for (day, hour), count in chunk.groupby(['day_of_week', 'hour']).size().items():
                key = (day, hour)
                hour_day_counts[key] = hour_day_counts.get(key, 0) + count

            # Histogram of message lengths and word counts.
            ml = chunk['message'].str.len()
            wc = chunk['message'].str.split().str.len()
            counts_ml, _ = np.histogram(ml, bins=ml_bins)
            ml_hist += counts_ml
            counts_wc, _ = np.histogram(wc, bins=wc_bins)
            wc_hist += counts_wc

            # Compute custom metrics (emote density, repetition ratio, unique word ratio)
            metrics = chunk['message'].apply(lambda m: compute_message_metrics(m, COMMON_EMOTES))
            chunk['emote_density'] = metrics.apply(lambda x: x[0])
            chunk['repetition_ratio'] = metrics.apply(lambda x: x[1])
            chunk['unique_ratio'] = metrics.apply(lambda x: x[2])
            all_emote_densities.extend(chunk['emote_density'].tolist())
            all_repetition_ratios.extend(chunk['repetition_ratio'].tolist())
            all_unique_ratios.extend(chunk['unique_ratio'].tolist())

            # Now sample a subset of rows (up to 1000) for scatter plots
            sample_size = min(1000, len(chunk))
            sample_chunk = chunk.sample(n=sample_size)
            scatter_lengths.extend(sample_chunk['message'].str.len().tolist())
            scatter_word_counts.extend(sample_chunk['message'].str.split().str.len().tolist())
            scatter_unique_ratios.extend(sample_chunk['unique_ratio'].tolist())

            # For box plots: collect a small sample of message lengths per channel.
            for channel, group in chunk.groupby('channel'):
                if len(group) > sample_per_channel_chunk:
                    group_sample = group.sample(n=sample_per_channel_chunk)
                else:
                    group_sample = group
                ml_values = group_sample['message'].str.len().tolist()
                if channel in channel_box_samples:
                    channel_box_samples[channel].extend(ml_values)
                    if len(channel_box_samples[channel]) > max_samples_per_channel:
                        channel_box_samples[channel] = random.sample(channel_box_samples[channel],
                                                                      max_samples_per_channel)
                else:
                    channel_box_samples[channel] = ml_values

            # Update word frequency counter.
            for msg in chunk['message'].dropna():
                words = msg.lower().split()
                word_freq.update(words)

            # Aggregate metrics by channel.
            for channel, group in chunk.groupby('channel'):
                ed_list = group['emote_density'].tolist()
                rr_list = group['repetition_ratio'].tolist()
                ur_list = group['unique_ratio'].tolist()
                if channel in channel_emote_densities:
                    channel_emote_densities[channel].extend(ed_list)
                    channel_repetition_ratios[channel].extend(rr_list)
                    channel_unique_ratios[channel].extend(ur_list)
                else:
                    channel_emote_densities[channel] = ed_list
                    channel_repetition_ratios[channel] = rr_list
                    channel_unique_ratios[channel] = ur_list

    return {
        'time_counts': time_counts,
        'channel_counts': channel_counts,
        'user_counts': user_counts,
        'hour_counts': hour_counts,
        'day_of_week_counts': day_of_week_counts,
        'hour_day_counts': hour_day_counts,
        'ml_bins': ml_bins,
        'ml_hist': ml_hist,
        'wc_bins': wc_bins,
        'wc_hist': wc_hist,
        'scatter_lengths': scatter_lengths,
        'scatter_word_counts': scatter_word_counts,
        'scatter_unique_ratios': scatter_unique_ratios,
        'channel_box_samples': channel_box_samples,
        'word_freq': word_freq,
        'all_emote_densities': all_emote_densities,
        'all_repetition_ratios': all_repetition_ratios,
        'all_unique_ratios': all_unique_ratios,
        'channel_emote_densities': channel_emote_densities,
        'channel_repetition_ratios': channel_repetition_ratios,
        'channel_unique_ratios': channel_unique_ratios
    }

def plot_graphs(agg):
    """
    Generate 20 different graphs using the aggregated data.
    Graphs are saved as PNG files.
    """
    # Graph 1: Time Series - Messages per Minute.
    time_series = pd.Series(agg['time_counts']).sort_index()
    plt.figure(figsize=(12,6))
    time_series.plot()
    plt.title('Twitch Chat Messages per Minute')
    plt.xlabel('Time')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    plt.savefig('graph1_time_series.png')
    plt.close()

    # Graph 2: Top 10 Channels by Message Count.
    channel_series = pd.Series(agg['channel_counts']).sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    channel_series.head(10).plot(kind='bar')
    plt.title('Top 10 Channels by Message Count')
    plt.xlabel('Channel')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    plt.savefig('graph2_top_channels.png')
    plt.close()

    # Graph 3: Histogram of Message Lengths (characters).
    plt.figure(figsize=(12,6))
    bins = agg['ml_bins']
    hist = agg['ml_hist']
    bin_labels = [f"{int(bins[i])}-{int(bins[i+1]-1) if bins[i+1]!=np.inf else '+'}" for i in range(len(bins)-1)]
    plt.bar(range(len(hist)), hist, tick_label=bin_labels)
    plt.title('Histogram of Message Lengths (characters)')
    plt.xlabel('Message Length')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('graph3_message_length_hist.png')
    plt.close()

    # Graph 4: Top 10 Active Users by Message Count.
    user_series = pd.Series(agg['user_counts']).sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    user_series.head(10).plot(kind='bar')
    plt.title('Top 10 Active Users by Message Count')
    plt.xlabel('Username')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    plt.savefig('graph4_top_users.png')
    plt.close()

    # Graph 5: Messages by Hour of Day.
    hour_series = pd.Series(agg['hour_counts']).sort_index()
    plt.figure(figsize=(12,6))
    hour_series.plot(kind='bar')
    plt.title('Messages by Hour of Day')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    plt.savefig('graph5_messages_by_hour.png')
    plt.close()

    # Graph 6: Messages by Day of Week.
    days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_series = pd.Series(agg['day_of_week_counts']).reindex(days_order)
    plt.figure(figsize=(12,6))
    day_series.plot(kind='bar')
    plt.title('Messages by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    plt.savefig('graph6_messages_by_day_of_week.png')
    plt.close()

    # Graph 7: Histogram of Word Counts per Message.
    plt.figure(figsize=(12,6))
    bins_wc = agg['wc_bins']
    hist_wc = agg['wc_hist']
    bin_labels_wc = [f"{int(bins_wc[i])}-{int(bins_wc[i+1]-1) if bins_wc[i+1]!=np.inf else '+'}" for i in range(len(bins_wc)-1)]
    plt.bar(range(len(hist_wc)), hist_wc, tick_label=bin_labels_wc)
    plt.title('Histogram of Word Counts per Message')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('graph7_word_count_hist.png')
    plt.close()

    # Graph 8: Scatter Plot - Message Length vs. Word Count.
    plt.figure(figsize=(12,6))
    plt.scatter(agg['scatter_lengths'], agg['scatter_word_counts'], alpha=0.5)
    plt.title('Scatter Plot: Message Length vs. Word Count')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Word Count')
    plt.tight_layout()
    plt.savefig('graph8_scatter_length_vs_wordcount.png')
    plt.close()

    # Graph 9: Box Plot of Message Lengths for Top 5 Channels.
    channel_series = pd.Series(agg['channel_counts']).sort_values(ascending=False)
    top5_channels = channel_series.head(5).index.tolist()
    box_data = [agg['channel_box_samples'][ch] for ch in top5_channels if ch in agg['channel_box_samples']]
    plt.figure(figsize=(12,6))
    plt.boxplot(box_data, tick_labels=top5_channels)  # use tick_labels instead of labels
    plt.title('Box Plot of Message Lengths for Top 5 Channels')
    plt.xlabel('Channel')
    plt.ylabel('Message Length (characters)')
    plt.tight_layout()
    plt.savefig('graph9_boxplot_top5_channels.png')
    plt.close()

    # Graph 10: Word Cloud of Frequently Used Words.
    try:
        from wordcloud import WordCloud
        wc_wordcloud = WordCloud(width=800, height=400, background_color='white') \
                        .generate_from_frequencies(agg['word_freq'])
        plt.figure(figsize=(12,6))
        plt.imshow(wc_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Frequently Used Words')
        plt.tight_layout()
        plt.savefig('graph10_wordcloud.png')
        plt.close()
    except ImportError:
        print("wordcloud package is not installed. Skipping word cloud generation.")

    # --- Additional 10 Graphs ---

    # Graph 11: Histogram of Emote Densities Across All Messages.
    plt.figure(figsize=(12,6))
    plt.hist(agg['all_emote_densities'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Emote Densities')
    plt.xlabel('Emote Density')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('graph11_emote_density_hist.png')
    plt.close()

    # Graph 12: Average Emote Density by Top 10 Channels.
    avg_emote = {ch: np.mean(densities) for ch, densities in agg['channel_emote_densities'].items() if densities}
    avg_emote_series = pd.Series(avg_emote).sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    avg_emote_series.head(10).plot(kind='bar', color='orange')
    plt.title('Average Emote Density by Top 10 Channels')
    plt.xlabel('Channel')
    plt.ylabel('Average Emote Density')
    plt.tight_layout()
    plt.savefig('graph12_avg_emote_density.png')
    plt.close()

    # Graph 13: Histogram of Repetition Ratios Across All Messages.
    plt.figure(figsize=(12,6))
    plt.hist(agg['all_repetition_ratios'], bins=20, color='lightgreen', edgecolor='black')
    plt.title('Histogram of Repetition Ratios')
    plt.xlabel('Repetition Ratio')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('graph13_repetition_ratio_hist.png')
    plt.close()

    # Graph 14: Average Repetition Ratio by Top 10 Channels.
    avg_repetition = {ch: np.mean(ratios) for ch, ratios in agg['channel_repetition_ratios'].items() if ratios}
    avg_repetition_series = pd.Series(avg_repetition).sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    avg_repetition_series.head(10).plot(kind='bar', color='red')
    plt.title('Average Repetition Ratio by Top 10 Channels')
    plt.xlabel('Channel')
    plt.ylabel('Average Repetition Ratio')
    plt.tight_layout()
    plt.savefig('graph14_avg_repetition_ratio.png')
    plt.close()

    # Graph 15: Scatter Plot - Emote Density vs. Repetition Ratio.
    plt.figure(figsize=(12,6))
    plt.scatter(agg['all_emote_densities'], agg['all_repetition_ratios'], alpha=0.5, color='purple')
    plt.title('Emote Density vs. Repetition Ratio')
    plt.xlabel('Emote Density')
    plt.ylabel('Repetition Ratio')
    plt.tight_layout()
    plt.savefig('graph15_emote_vs_repetition.png')
    plt.close()

    # Graph 16: Top 20 Most Common Words (Bar Chart).
    top_words = dict(Counter(agg['word_freq']).most_common(20))
    plt.figure(figsize=(12,6))
    pd.Series(top_words).plot(kind='bar', color='teal')
    plt.title('Top 20 Most Common Words')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('graph16_top_words.png')
    plt.close()

    # Graph 17: Histogram of Unique Word Ratios Across All Messages.
    plt.figure(figsize=(12,6))
    plt.hist(agg['all_unique_ratios'], bins=20, color='pink', edgecolor='black')
    plt.title('Histogram of Unique Word Ratios')
    plt.xlabel('Unique Word Ratio')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('graph17_unique_word_ratio_hist.png')
    plt.close()

    # Graph 18: Scatter Plot - Message Length vs. Unique Word Ratio.
    plt.figure(figsize=(12,6))
    plt.scatter(agg['scatter_lengths'], agg['scatter_unique_ratios'], alpha=0.5, color='brown')
    plt.title('Message Length vs. Unique Word Ratio')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Unique Word Ratio')
    plt.tight_layout()
    plt.savefig('graph18_length_vs_unique_ratio.png')
    plt.close()

    # Graph 19: Heatmap of Messages by Day of Week and Hour.
    heatmap_data = pd.DataFrame(0, index=days_order, columns=range(24))
    for (day, hour), count in agg['hour_day_counts'].items():
        if day in heatmap_data.index and hour in heatmap_data.columns:
            heatmap_data.at[day, hour] = count
    plt.figure(figsize=(12,6))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Message Count')
    plt.xticks(ticks=range(24), labels=range(24))
    plt.yticks(ticks=range(len(days_order)), labels=days_order)
    plt.title('Heatmap: Messages by Day of Week and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig('graph19_day_hour_heatmap.png')
    plt.close()

    # Graph 20: Pie Chart - Distribution of Messages Across Top 5 Channels.
    top5 = channel_series.head(5)
    plt.figure(figsize=(8,8))
    plt.pie(top5, labels=top5.index, autopct='%1.1f%%', startangle=140)
    plt.title('Message Distribution Among Top 5 Channels')
    plt.tight_layout()
    plt.savefig('graph20_top5_channels_pie.png')
    plt.close()

def main():
    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
    else:
        file_paths = glob.glob("*.csv")
        if not file_paths:
            print("No CSV files provided or found in the current directory.")
            sys.exit(1)
    
    print("Processing CSV files...")
    aggregated = process_csv_files(file_paths, chunksize=10**6)
    print("CSV processing complete. Generating graphs...")
    plot_graphs(aggregated)
    print("Graphs saved as PNG files (graph1_*.png to graph20_*.png).")

if __name__ == '__main__':
    main()
