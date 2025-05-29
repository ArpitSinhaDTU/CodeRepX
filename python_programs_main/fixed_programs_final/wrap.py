def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = text.rfind(' ', 0, cols + 1)
        if end == -1 or end == 0:
            end = cols
        line, text = text[:end], text[end:]
        lines.append(line)
    if text:
        lines.append(text)
    return lines