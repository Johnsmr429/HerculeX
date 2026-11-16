(() => {
  const GROUPS = {
    move: "Move",
    look: "Look",
    places: "Places",
    ink: "Ink and meshes",
    label: "Label",
  };
  // [group, keys (as event.key lowercased), action, what it does, may repeat while held]
  const KEYS = [
    ["move", ["w", "a", "s", "d"], "move", "Move (hold)"],
    ["move", ["e", "r"], "move", "Zoom out, zoom in (hold)"],
    ["move", ["arrowleft", "arrowright"], "panLeft|panRight", "Move left, right a screen"],
    ["move", ["pageup", "pagedown"], "panUp|panDown", "Move up, down a screen"],
    ["move", [" "], "grab", "Hold and drag to move"],
    ["move", ["0"], "voxel", "One voxel per pixel"],
    ["look", ["arrowup", "arrowdown"], "layerUp|layerDown", "Next layer, previous layer"],
    ["look", [",", "."], "floorDown|floorUp", "Weakest ink shown", true],
    ["look", [";", "'"], "scanDown|scanUp", "Brightness of the scan under the ink", true],
    ["look", ["h"], "mirror", "Mirror"],
    ["look", ["u"], "turn", "Turn"],
    ["look", ["b"], "boxes", "Hide or show boxes"],
    ["look", ["k"], "greek", "Greek alphabet in real letters of the scrolls"],
    ["look", ["f"], "full", "Full screen"],
    ["look", ["escape"], "back", "Back"],
    ["look", ["?"], "help", "This list"],
    ["places", ["n", "p"], "next|previous", "Next, previous place"],
    ["places", ["j"], "suggest", "Next suggested area or square (Shift: previous)"],
    ["ink", ["t"], "swap", "Swap the two inks or meshes"],
    ["ink", ["y"], "yellow", "Yellow ink"],
    ["ink", ["i"], "heat", "Ink in colours by its strength"],
    ["ink", ["1", "2"], "meshA|meshB", "First mesh, new mesh on or off"],
    ["ink", ["3", "4"], "inkY|inkP", "First ink, new ink on or off"],
    ["ink", ["q"], "squares", "Squares on or off"],
    ["label", ["g", "m"], "mark", "Marker on or off"],
    ["label", ["c"], "spot", "Circle spots on or off"],
    ["label", ["x"], "erase", "Eraser on or off"],
    ["label", ["z"], "undo", "Undo"],
    ["label", ["l"], "marks", "Hide or show marks"],
    ["label", ["[", "]"], "brushDown|brushUp", "Marker width", true],
  ];
  const HELD = ["w", "a", "s", "d", "e", "r"];
  const map = new Map();
  for (const [group, keys, action, label, repeats] of KEYS) {
    const names = action.split("|");
    keys.forEach((key, i) => map.set(key, { group, action: names[i] ?? names[0], label, repeats, all: action }));
  }
  const held = new Set();
  let actions = {}, panel = null;

  const has = (all) => all.split("|").some((name) => name === "help" || actions[name]);
  const show = (key) => ({ " ": "Space", arrowleft: "←", arrowright: "→", arrowup: "↑", arrowdown: "↓", pageup: "Page Up", pagedown: "Page Down", escape: "Esc" })[key] ?? key.toUpperCase();

  function help(open) {
    if (!panel) {
      panel = document.createElement("div");
      panel.hidden = true;
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-label", "Keyboard shortcuts");
      panel.style.cssText = "position:fixed;inset:0;z-index:20;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.82);color:#bbb;font:10pt Calibri,Carlito,sans-serif";
      panel.onclick = () => help(false);
      document.body.append(panel);
    }
    open = open ?? panel.hidden;
    if (open) {
      const seen = new Set(), sections = {};
      for (const [group, keys, action, label] of KEYS) {
        if (!has(action) || seen.has(label)) continue;
        seen.add(label);
        const caps = keys.map((key) => `<kbd style="display:inline-block;min-width:1.6em;padding:1px 6px;margin-right:4px;text-align:center;border:1px solid #444;border-radius:3px;background:#111;color:#fff;font:inherit">${show(key)}</kbd>`).join("");
        (sections[group] ??= []).push(`<div style="display:flex;gap:14px;justify-content:space-between;padding:2px 0"><span style="white-space:nowrap">${caps}</span><span>${label}</span></div>`);
      }
      panel.innerHTML = `<div style="max-width:min(1100px,92vw);max-height:90vh;overflow:auto;columns:2 380px;column-gap:48px">${Object.keys(GROUPS)
        .filter((group) => sections[group])
        .map((group) => `<div style="break-inside:avoid;margin-bottom:16px"><div style="color:#fff;margin-bottom:4px">${GROUPS[group]}</div>${sections[group].join("")}</div>`)
        .join("")}</div>`;
    }
    panel.hidden = !open;
  }

  addEventListener("keydown", (e) => {
    if (e.key === "Escape" && panel && !panel.hidden) {
      help(false);
      e.stopImmediatePropagation();
      return;
    }
    if (e.target.matches("input,select,textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const key = e.key.toLowerCase();
    const entry = map.get(key);
    if (!entry) return;
    if (HELD.includes(key)) {
      if (!actions.move) return;
      if (!held.size) actions.start?.();
      held.add(key);
      actions.wake?.();
      return;
    }
    const run = entry.action === "help" ? () => help() : actions[entry.action];
    if (!run) return;
    if (e.repeat && !entry.repeats) return;
    if (key === " " || key.startsWith("arrow") || key.startsWith("page")) e.preventDefault();
    run(e);
  });
  addEventListener("keyup", (e) => {
    const key = e.key.toLowerCase();
    held.delete(key);
    const entry = map.get(key);
    if (entry) actions[entry.action + "End"]?.(e);
  });
  addEventListener("blur", () => held.clear());

  window.Shortcuts = { held, help, bind: (given) => (actions = given) };
})();
