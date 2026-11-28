function spotReadout(output, scan, toView) {
    const [millimetres, source, pixels] = output.children;
    source.textContent = `scan=${scan}`;
    const cached = new WeakMap();

    function bounds(points) {
        let left = Infinity,
            top = Infinity,
            right = -Infinity,
            bottom = -Infinity;
        for (const [x, y] of points) {
            left = Math.min(left, x);
            top = Math.min(top, y);
            right = Math.max(right, x);
            bottom = Math.max(bottom, y);
        }
        return [left, top, right, bottom];
    }

    function screenPixel(value, offset, zoom) {
        return Math.round((value - offset) * zoom * 1e6) / 1e6;
    }

    function screenSpan(first, last, offset, zoom, flipped, size) {
        const a = screenPixel(first, offset, zoom),
            b = screenPixel(last, offset, zoom);
        return flipped ? [size - b, size - a] : [a, b];
    }

    return (marks, seen, { ox, oy, zoom, flipX, flipY }) => {
        const width = innerWidth,
            height = innerHeight;
        for (let i = marks.length - 1; i >= 0; i--) {
            const mark = marks[i];
            if (mark.kind !== "spot" || !seen(mark)) {
                continue;
            }
            if (!cached.has(mark)) {
                const [x0, y0, x1, y1] = bounds(mark.points);
                cached.set(mark, {
                    sheet: `XYmm ${x0},${y0} ${x1},${y1}`,
                    view: bounds(mark.points.map(toView)),
                });
            }
            const { sheet, view } = cached.get(mark);
            const [left, right] = screenSpan(view[0], view[2], ox, zoom, flipX, width);
            const [top, bottom] = screenSpan(view[1], view[3], oy, zoom, flipY, height);
            if (right < 0 || bottom < 0 || left >= width || top >= height) {
                continue;
            }

            const x0 = Math.max(0, Math.floor(left)),
                y0 = Math.max(0, Math.floor(top));
            const x1 = Math.min(width, Math.max(x0 + 1, Math.ceil(right)));
            const y1 = Math.min(height, Math.max(y0 + 1, Math.ceil(bottom)));
            const px = `ROI px=${x0},${y0},${x1},${y1}/${width}x${height}`;
            if (millimetres.textContent !== sheet) {
                millimetres.textContent = sheet;
            }
            if (pixels.textContent !== px) {
                pixels.textContent = px;
            }
            output.hidden = false;
            return;
        }
        output.hidden = true;
    };
}
