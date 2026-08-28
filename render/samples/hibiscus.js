async function setup() {
  createCanvas(600, 600, WEBGL);
  angleMode(DEGREES);
  brush.scaleBrushes(3);
}

function draw() {
  translate(-width / 2, -height / 2);

  // Buttery yellow / tea-stained background
  background("#f4e8c1");

  // Faint clinical field journal grid
  brush.set("2H", "#7a7465", 0.15);
  for (let x = 40; x <= 560; x += 40) {
    brush.line(x, 40, x, 560);
  }
  for (let y = 40; y <= 560; y += 40) {
    brush.line(40, y, 560, y);
  }

  // Antique water stains
  brush.noStroke();
  brush.fillBleed(0.4, "out");
  brush.fillTexture(0.85, 0.9);

  brush.fill("#9b7b5a", 30);
  brush.circle(130, 150, 75, 0.2);
  brush.circle(480, 420, 95, 0.3);
  brush.circle(410, 110, 45, 0.15);

  brush.fill("#8b6b4a", 25);
  brush.circle(280, 490, 50, 0.1);
  brush.circle(170, 380, 30, 0.1);

  // Darkened, ragged parchment edges
  brush.fill("#4a3520", 40);
  brush.fillBleed(0.5, "in");
  brush.field("hand");
  brush.rect(300, 8, 600, 16, "center");
  brush.rect(300, 592, 600, 16, "center");
  brush.rect(8, 300, 16, 600, "center");
  brush.rect(592, 300, 16, 600, "center");
  brush.noField();

  // Setup diagrammatic hatching properties
  brush.hatchStyle("HB", "#2a2a2a", 0.4);

  // Olive drab stem and sepals
  brush.set("HB", "#333333", 0.4);
  brush.fill("#6b705c", 50);
  brush.fillBleed(0.2, "out");
  brush.hatch(6, 45, { rand: 0.1 });

  brush.beginShape(0.1);
  brush.vertex(296, 290);
  brush.vertex(304, 290);
  brush.vertex(302, 450);
  brush.vertex(298, 450);
  brush.endShape(true);

  // Receptacle & Sepals
  brush.hatch(5, -45, { rand: 0.1 });
  brush.beginShape(0.2);
  brush.vertex(275, 275);
  brush.vertex(325, 275);
  brush.vertex(340, 230);
  brush.vertex(315, 260);
  brush.vertex(300, 295);
  brush.vertex(285, 260);
  brush.vertex(260, 230);
  brush.endShape(true);

  // Half-open Bud Petals
  brush.fillBleed(0.15, "out");

  brush.fill("#c68e8c", 45);
  brush.hatch(4, 15, { rand: 0.2 });
  brush.beginShape(0.5);
  brush.vertex(280, 275);
  brush.vertex(250, 180);
  brush.vertex(265, 110);
  brush.vertex(305, 155);
  brush.vertex(300, 275);
  brush.endShape(true);

  brush.fill("#c06c5a", 40);
  brush.hatch(5, -20, { rand: 0.15 });
  brush.beginShape(0.4);
  brush.vertex(300, 275);
  brush.vertex(315, 150);
  brush.vertex(345, 115);
  brush.vertex(365, 195);
  brush.vertex(320, 275);
  brush.endShape(true);

  brush.fill("#b66a5e", 55);
  brush.hatch(7, 75, { rand: 0.1 });
  brush.beginShape(0.3);
  brush.vertex(290, 275);
  brush.vertex(280, 195);
  brush.vertex(305, 130);
  brush.vertex(335, 180);
  brush.vertex(310, 275);
  brush.endShape(true);

  // Technical lines / Clinical markup
  brush.noFill();
  brush.noHatch();
  brush.set("2H", "#2a2a2a", 0.35);

  brush.line(265, 130, 110, 90);
  brush.circle(110, 90, 2);
  brush.line(110, 90, 70, 90);

  brush.line(345, 140, 480, 95);
  brush.circle(480, 95, 2);
  brush.line(480, 95, 530, 95);

  // Specimen scale bar
  brush.line(60, 540, 160, 540);
  brush.line(60, 535, 60, 545);
  brush.line(110, 537, 110, 543);
  brush.line(160, 535, 160, 545);

  noLoop();
}
