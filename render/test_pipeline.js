// Exercises the render pipeline: valid renders (warm timing) + failure capture.
const fs = require('fs');
const path = require('path');
const { renderSketch, closeBrowser } = require('./render');

const goodA = fs.readFileSync(path.join(__dirname, 'samples', 'hibiscus.js'), 'utf8');

const goodB = `
function setup(){ createCanvas(400,400,WEBGL); angleMode(DEGREES); brush.scaleBrushes(2); }
function draw(){
  translate(-width/2,-height/2);
  background("#eef3f7");
  brush.set("HB","#2b5d8a",0.9);
  brush.fill("#4a90d9",60); brush.fillBleed(0.3,"out");
  brush.circle(200,200,120,0.4);
  brush.noFill(); brush.set("charcoal","#123",0.7);
  brush.line(60,340,340,340);
  noLoop();
}`;

const syntaxErr = `function setup(){ createCanvas(300,300,WEBGL); }
function draw(){ background("#fff"); brush.line(0,0,10,10  // <- missing paren
  noLoop(); }`;

const runtimeErr = `function setup(){ createCanvas(300,300,WEBGL); }
function draw(){ background("#fff"); nonexistentThing.doStuff(); noLoop(); }`;

const noDraw = `function setup(){ createCanvas(300,300,WEBGL); }`;

(async () => {
  const cases = [
    ['good/hibiscus', goodA, { outPath: 'out/test_hibiscus.png' }],
    ['good/koi-ish',  goodB, { outPath: 'out/test_blue.png', width:400, height:400 }],
    ['good/warm2',    goodB, { outPath: 'out/test_blue2.png', width:400, height:400 }],
    ['bad/syntax',    syntaxErr, { timeoutMs: 8000 }],
    ['bad/runtime',   runtimeErr, { timeoutMs: 8000 }],
    ['bad/no-draw',   noDraw, { timeoutMs: 8000 }],
  ];
  for (const [name, code, opts] of cases) {
    const r = await renderSketch(code, opts);
    if (r.ok) {
      console.log(`  PASS  ${name.padEnd(16)} ok=true  ${r.width}x${r.height}  ${r.ms}ms  -> ${r.outPath}`);
    } else {
      const first = String(r.error).split('\n')[0].slice(0,90);
      console.log(`  GATE  ${name.padEnd(16)} ok=false ${r.ms}ms  err="${first}"`);
    }
  }
  await closeBrowser();
})();
