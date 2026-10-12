"""Complete-utterance window helpers from the existing meeting preparation."""
RATE = 16000


def complete_windows(samples, annotations, target_seconds):
    """Cut at silence/turn boundaries; never crop an annotated utterance."""
    intervals = sorted((round(row['start'] * RATE), round((row['start'] + row['duration']) * RATE))
                       for row in annotations)
    clusters = []
    for start, end in intervals:
        if clusters and start < clusters[-1][1]:
            clusters[-1][1] = max(clusters[-1][1], end)
        else:
            clusters.append([start, end])
    left, index = 0, 0
    while left < samples:
        right = min(left + round(target_seconds * RATE), samples)
        while index < len(clusters) and clusters[index][1] <= right:
            index += 1
        if index < len(clusters):
            start, end = clusters[index]
            if start < right < end:
                right = start if start > left else end
        assert left < right <= samples
        yield left, right
        left = right


def render_target(annotations, offset):
    mapping, turns, lines = {}, [], []
    for row in sorted(annotations, key=lambda r: (r['start'], r['speaker'], r['id'])):
        label = mapping.setdefault(row['speaker'], f'S{len(mapping) + 1}')
        start = round(row['start'] * RATE) / RATE - offset
        end = round((row['start'] + row['duration']) * RATE) / RATE - offset
        text = row['text'].strip()
        precision = 6 if round(start, 2) >= round(end, 2) else 2
        lines.append(f'[{label}][{start:.{precision}f}-{end:.{precision}f}] {text}')
        turns.append({'source_id': row['id'], 'speaker': label,
                      'source_speaker': row['speaker'], 'start': start, 'end': end, 'text': text})
    return '\n'.join(lines), turns, mapping
