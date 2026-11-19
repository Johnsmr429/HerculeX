function spotReadout(output, scan, toView) {
  const [millimetres, source, pixels] = output.children;
  source.textContent = "scan=" + scan;
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
    // Adding and subtracting a shelf offset can put an integer just over a pixel boundary.
    return Math.round((value - offset) * zoom * 1e6) / 1e6;
  }

  return (marks, seen, { ox, oy, zoom, flipX, flipY }) => {
    const width = innerWidth,
      height = innerHeight;
    for (let i = marks.length - 1; i >= 0; i--) {
      const mark = marks[i];
      if (mark.kind !== "spot" || !seen(mark)) continue;
      if (!cached.has(mark)) {
        const [x0, y0, x1, y1] = bounds(mark.points);
        cached.set(mark, {
          sheet: `XYmm ${x0},${y0} ${x1},${y1}`,
          view: bounds(mark.points.map(toView)),
        });
      }
      const { sheet, view } = cached.get(mark);
      let left = screenPixel(view[0], ox, zoom),
        right = screenPixel(view[2], ox, zoom);
      let top = screenPixel(view[1], oy, zoom),
        bottom = screenPixel(view[3], oy, zoom);
      if (flipX) [left, right] = [width - right, width - left];
      if (flipY) [top, bottom] = [height - bottom, height - top];
      if (right < 0 || bottom < 0 || left >= width || top >= height) continue;

      // Half-open viewport bounds also give a one-pixel crop for a single click.
      const x0 = Math.max(0, Math.floor(left)),
        y0 = Math.max(0, Math.floor(top));
      const x1 = Math.min(width, Math.max(x0 + 1, Math.ceil(right)));
      const y1 = Math.min(height, Math.max(y0 + 1, Math.ceil(bottom)));
      const px = `ROI px=${x0},${y0},${x1},${y1}/${width}x${height}`;
      if (millimetres.textContent !== sheet) millimetres.textContent = sheet;
      if (pixels.textContent !== px) pixels.textContent = px;
      output.hidden = false;
      return;
    }
    output.hidden = true;
  };
}
