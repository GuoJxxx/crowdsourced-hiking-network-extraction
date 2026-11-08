"""Read current-stage work counts emitted by a running reconstruction method."""

import json


def parse_progress(line):
    if not line.startswith('[PROGRESS] '):
        return None
    try:
        record = json.loads(line[len('[PROGRESS] '):])
        if not isinstance(record, dict):
            return None
        phase, completed, total = (record.get(key) for key in ('phase', 'completed', 'total'))
        if (not isinstance(phase, str) or not phase or len(phase) > 120
                or any(not 32 <= ord(c) <= 126 for c in phase)
                or type(completed) is not int or type(total) is not int
                or not 0 <= completed <= total):
            return None
        percent = completed*100//total if total else 100
        return {'phase': phase, 'completed': completed, 'total': total, 'percent': percent}
    except (ValueError, TypeError):
        return None
