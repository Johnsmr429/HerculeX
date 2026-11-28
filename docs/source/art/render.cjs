const fs = require("node:fs");
const path = require("node:path");
const { createCanvas } = require("@napi-rs/canvas");
const Minoan = require("./minoan.js");

const images = path.resolve(__dirname, "../../images");
const jobs = process.argv[2]
  ? [[process.argv[2], process.argv[3] || path.join(images, `herculex-${process.argv[2]}.png`)]]
  : [
      ["hero", path.join(images, "herculex-hero.png")],
      ["social", path.join(images, "herculex-social.png")],
      ["banner", path.resolve(__dirname, "../../../viewer/assets/herculex-banner.jpg")],
    ];

for (const [name, file] of jobs) {
  const [w, h] = Minoan.SCENES[name].size;
  const canvas = createCanvas(w, h);
  const started = Date.now();
  Minoan.paint(canvas.getContext("2d"), name, createCanvas);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, canvas.toBuffer("image/png"));
  const kb = Math.round(fs.statSync(file).size / 1024);
  console.log(`${name}: ${w} x ${h} -> ${path.relative(process.cwd(), file)} (${kb} KB, ${Date.now() - started} ms)`);
}
