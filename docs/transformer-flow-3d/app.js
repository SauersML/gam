import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

const COLORS = {
  background: 0x08090d,
  graphite: 0xc9c8ce,
  violet: 0x9d79ff,
  violetBright: 0xc7b4ff,
  cyan: 0x28cfff,
  magenta: 0xf14baa,
  orange: 0xff9f43,
};

const TOKENS = ["You", "are", "absolutely", "correct", "next token"];
const LAYERS = 4;
const X_SPACING = 3.05;
const Y_SPACING = 2.65;
const xAt = (position) => (position - (TOKENS.length - 1) / 2) * X_SPACING;
const yAt = (layer) => (layer - (LAYERS - 1) / 2) * Y_SPACING;
const urlParameters = new URLSearchParams(window.location.search);
if (urlParameters.get("stage") === "1") document.body.classList.add("stage-only");

const mount = document.querySelector("#scene");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(mount.clientWidth, mount.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.setClearColor(COLORS.background, 0);
mount.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(COLORS.background, 0.027);

const camera = new THREE.PerspectiveCamera(35, mount.clientWidth / mount.clientHeight, 0.1, 100);
camera.position.set(10.8, 5.3, 18.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.055;
controls.enablePan = false;
controls.minDistance = 6;
controls.maxDistance = 32;
controls.maxPolarAngle = Math.PI * 0.82;
controls.target.set(0, 0, 0);

scene.add(new THREE.HemisphereLight(0xb8c8ff, 0x100b1f, 1.35));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
keyLight.position.set(5, 9, 12);
scene.add(keyLight);
const violetLight = new THREE.PointLight(COLORS.violet, 22, 22, 2);
violetLight.position.set(-4, 1, 5);
scene.add(violetLight);
const cyanLight = new THREE.PointLight(COLORS.cyan, 12, 18, 2);
cyanLight.position.set(6, -1, 4);
scene.add(cyanLight);

const systemGroup = new THREE.Group();
const residualGroup = new THREE.Group();
const connectionGroup = new THREE.Group();
const moduleGroup = new THREE.Group();
const labelGroup = new THREE.Group();
const pathGroup = new THREE.Group();
systemGroup.add(residualGroup, connectionGroup, moduleGroup, labelGroup);
scene.add(systemGroup, pathGroup);

const pickables = [];
const residuals = [];
const connections = [];
const modules = [];
const flowParticles = [];

function physicalMaterial(color, options = {}) {
  return new THREE.MeshPhysicalMaterial({
    color,
    emissive: color,
    emissiveIntensity: options.emissiveIntensity ?? 0.5,
    roughness: options.roughness ?? 0.25,
    metalness: options.metalness ?? 0.2,
    transparent: true,
    opacity: options.opacity ?? 1,
    clearcoat: options.clearcoat ?? 0.7,
    clearcoatRoughness: 0.28,
    depthWrite: options.depthWrite ?? true,
  });
}

function basicMaterial(color, opacity = 1) {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: opacity > 0.45,
    blending: opacity < 0.5 ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
}

function tubeFromCurve(curve, color, radius, opacity, segments = 48) {
  const geometry = new THREE.TubeGeometry(curve, segments, radius, 7, false);
  const mesh = new THREE.Mesh(geometry, basicMaterial(color, opacity));
  mesh.userData.curve = curve;
  mesh.userData.baseOpacity = opacity;
  return mesh;
}

function lineCurve(a, b) {
  return new THREE.LineCurve3(a, b);
}

function addArrow(position, direction, color, scale = 1, opacity = 1) {
  const arrow = new THREE.Mesh(
    new THREE.ConeGeometry(0.11 * scale, 0.34 * scale, 10),
    basicMaterial(color, opacity),
  );
  arrow.position.copy(position);
  arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
  arrow.userData.baseOpacity = opacity;
  return arrow;
}

function createTextSprite(text, color = "#8a8892", fontSize = 34) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 512;
  canvas.height = 96;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = `500 ${fontSize}px DM Mono, monospace`;
  context.fillStyle = color;
  context.letterSpacing = "4px";
  context.fillText(text, 12, 58);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.72, depthWrite: false }));
  sprite.scale.set(3.55, 0.66, 1);
  sprite.userData.baseOpacity = 0.72;
  return sprite;
}

function addParticle(curve, color, radius, offset, speed, modes) {
  const particle = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 14, 10),
    physicalMaterial(color, { emissiveIntensity: 3.4, roughness: 0.1, opacity: 1 }),
  );
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(radius * 2.8, 12, 8),
    basicMaterial(color, 0.13),
  );
  particle.add(glow);
  particle.userData.baseOpacity = 1;
  particle.userData.flow = { curve, offset, speed, modes };
  scene.add(particle);
  flowParticles.push(particle);
  return particle;
}

function tagPickable(mesh, data) {
  mesh.userData.info = data;
  pickables.push(mesh);
}

function createResidualColumns() {
  const yMin = yAt(0) - 1.35;
  const yMax = yAt(LAYERS - 1) + 1.45;

  for (let position = 0; position < TOKENS.length; position += 1) {
    const x = xAt(position);
    const curve = lineCurve(new THREE.Vector3(x, yMin, 0), new THREE.Vector3(x, yMax, 0));
    const halo = tubeFromCurve(curve, COLORS.graphite, 0.105, 0.055, 80);
    const core = tubeFromCurve(curve, COLORS.graphite, 0.035, 0.58, 80);
    halo.userData.kind = "residual";
    core.userData.kind = "residual";
    core.userData.position = position;
    residualGroup.add(halo, core);
    residuals.push(halo, core);
    tagPickable(core, {
      id: `RES / T${position}`,
      kicker: "VERTICAL HIGHWAY",
      title: "Residual state carries the <em>working state upward.</em>",
      description: `At “${TOKENS[position]}”, the hidden state persists through depth. Attention and MLP updates are added into this stream rather than replacing it. Its width—the model dimension—is the central information bottleneck.`,
      color: "#d7d5dc",
    });

    for (let layer = 0; layer < LAYERS; layer += 1) {
      const arrow = addArrow(new THREE.Vector3(x, yAt(layer) + 1.18, 0), new THREE.Vector3(0, 1, 0), COLORS.graphite, 0.72, 0.66);
      arrow.userData.kind = "residual";
      arrow.userData.position = position;
      residualGroup.add(arrow);
      residuals.push(arrow);
    }

    addParticle(curve, COLORS.graphite, 0.065, position / TOKENS.length, 0.09, ["overview", "block"]);
  }
}

function createModule(layer, position) {
  const x = xAt(position);
  const y = yAt(layer);
  const group = new THREE.Group();
  group.position.set(x, y, 0);
  group.userData.layer = layer;
  group.userData.position = position;

  const kvStemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, -0.9, 0.02),
    new THREE.Vector3(0, -0.68, 0.24),
    new THREE.Vector3(0, -0.55, 0.58),
  ]);
  const kvStem = tubeFromCurve(kvStemCurve, COLORS.violet, 0.045, 0.82, 16);
  kvStem.userData.kind = "module";
  group.add(kvStem);

  const kv = new THREE.Mesh(new THREE.SphereGeometry(0.19, 24, 18), physicalMaterial(COLORS.violet, { emissiveIntensity: 1.55 }));
  kv.position.set(0, -0.51, 0.62);
  kv.userData.kind = "module";
  group.add(kv);
  modules.push(kv, kvStem);
  tagPickable(kv, {
    id: `K/V · L${layer + 1} T${position}`,
    kicker: "KEY / VALUE PROJECTION",
    title: "This state becomes a <em>readable memory.</em>",
    description: `At layer ${layer + 1}, position “${TOKENS[position]}” projects its normalized residual state into keys and values. Keys determine future relevance; values carry the content retrieved by matching queries.`,
    color: "#a78bfa",
  });

  const localKVRead = tubeFromCurve(
    new THREE.CubicBezierCurve3(
      new THREE.Vector3(0, -0.51, 0.62),
      new THREE.Vector3(0.28, -0.34, 0.84),
      new THREE.Vector3(0.28, -0.08, 0.84),
      new THREE.Vector3(0.08, 0.02, 0.73),
    ),
    COLORS.violet,
    0.026,
    0.78,
    18,
  );
  localKVRead.userData.kind = "module";
  group.add(localKVRead);
  modules.push(localKVRead);

  const attention = new THREE.Mesh(
    new RoundedBoxGeometry(0.86, 0.54, 0.72, 5, 0.09),
    physicalMaterial(COLORS.cyan, { emissiveIntensity: 0.62, roughness: 0.18 }),
  );
  attention.position.set(0, 0.02, 0.36);
  attention.userData.kind = "module";
  group.add(attention);
  modules.push(attention);
  tagPickable(attention, {
    id: `ATTN · L${layer + 1} T${position}`,
    kicker: "CAUSAL SELF-ATTENTION",
    title: "A query turns context into a <em>weighted read.</em>",
    description: `The query at “${TOKENS[position]}” scores keys from positions 0…${position}. Softmax converts those scores to weights, and the weighted value sum is projected back into the residual width. No future token is visible.`,
    color: "#35d6ff",
  });

  const attnOutput = tubeFromCurve(
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.29, 0.36),
      new THREE.Vector3(0, 0.42, 0.25),
      new THREE.Vector3(0, 0.5, 0.04),
    ]),
    COLORS.cyan,
    0.04,
    0.82,
    14,
  );
  attnOutput.userData.kind = "module";
  group.add(attnOutput);
  modules.push(attnOutput);

  const addOne = new THREE.Mesh(
    new THREE.TorusGeometry(0.15, 0.03, 10, 24),
    physicalMaterial(COLORS.graphite, { emissiveIntensity: 0.25, roughness: 0.3 }),
  );
  addOne.rotation.x = Math.PI / 2;
  addOne.position.set(0, 0.5, 0.02);
  addOne.userData.kind = "module";
  group.add(addOne);
  modules.push(addOne);

  const mlp = new THREE.Mesh(
    new RoundedBoxGeometry(0.98, 0.5, 0.67, 5, 0.09),
    physicalMaterial(COLORS.magenta, { emissiveIntensity: 0.56, roughness: 0.2 }),
  );
  mlp.position.set(0, 0.88, 0.3);
  mlp.userData.kind = "module";
  group.add(mlp);
  modules.push(mlp);
  tagPickable(mlp, {
    id: `MLP · L${layer + 1} T${position}`,
    kicker: "POSITION-WISE MLP",
    title: "The retrieved context is <em>transformed locally.</em>",
    description: `After attention is added to the residual stream, the MLP transforms the state at “${TOKENS[position]}” independently of other positions. Its output is added through a second residual connection.`,
    color: "#f34bb5",
  });

  const topStem = tubeFromCurve(
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 1.12, 0.3),
      new THREE.Vector3(0, 1.2, 0.12),
      new THREE.Vector3(0, 1.28, 0),
    ]),
    COLORS.magenta,
    0.04,
    0.75,
    12,
  );
  topStem.userData.kind = "module";
  group.add(topStem);
  modules.push(topStem);

  moduleGroup.add(group);
}

function createKVConnections() {
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const y = yAt(layer);
    for (let target = 0; target < TOKENS.length; target += 1) {
      for (let source = 0; source <= target; source += 1) {
        if (source === target) continue;
        const distance = target - source;
        const start = new THREE.Vector3(xAt(source), y - 0.51, 0.62);
        const end = new THREE.Vector3(xAt(target) - 0.06, y + 0.02, 0.75);
        const zLane = 0.9 + distance * 0.25;
        const sag = 0.25 + distance * 0.055;
        const curve = new THREE.CubicBezierCurve3(
          start,
          new THREE.Vector3(xAt(source) + distance * 0.75, y - sag, zLane),
          new THREE.Vector3(xAt(target) - distance * 0.68, y - sag, zLane),
          end,
        );
        const connection = tubeFromCurve(curve, COLORS.violet, 0.027, 0.105, 34 + distance * 7);
        connection.userData.kind = "connection";
        connection.userData.layer = layer;
        connection.userData.source = source;
        connection.userData.target = target;
        connection.userData.focus = target === 3 && layer === 2;
        connectionGroup.add(connection);
        connections.push(connection);
        tagPickable(connection, {
          id: `READ · L${layer + 1} T${source}→T${target}`,
          kicker: "K/V RETRIEVAL EDGE",
          title: "A later position can <em>retrieve this value.</em>",
          description: `At layer ${layer + 1}, the query at “${TOKENS[target]}” compares against the key stored at “${TOKENS[source]}”. Its attention weight controls how much of that value enters the current computation.`,
          color: "#a78bfa",
        });

        if (connection.userData.focus) {
          connection.material.opacity = 0.72;
          connection.userData.baseOpacity = 0.72;
          const tangent = curve.getTangent(0.78).normalize();
          const arrow = addArrow(curve.getPoint(0.78), tangent, COLORS.violetBright, 0.66, 0.8);
          arrow.userData.kind = "connection";
          arrow.userData.layer = layer;
          arrow.userData.source = source;
          arrow.userData.target = target;
          connectionGroup.add(arrow);
          connections.push(arrow);
          addParticle(curve, COLORS.violetBright, 0.07, source * 0.21, 0.13 + distance * 0.012, ["overview", "block"]);
        }
      }
    }
  }
}

function createLayerLabels() {
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const sprite = createTextSprite(`LAYER  ${String(layer + 1).padStart(2, "0")}`);
    sprite.position.set(xAt(0) - 1.45, yAt(layer), -0.2);
    labelGroup.add(sprite);
  }
}

function combinations(stepsRight, stepsUp) {
  const paths = [];
  function walk(right, up, current) {
    if (right === stepsRight && up === stepsUp) {
      paths.push(current);
      return;
    }
    if (right < stepsRight) walk(right + 1, up, `${current}R`);
    if (up < stepsUp) walk(right, up + 1, `${current}U`);
  }
  walk(0, 0, "");
  return paths;
}

function createCausalPaths() {
  const paths = combinations(3, 3);
  paths.forEach((path, pathIndex) => {
    let position = 0;
    let layer = 0;
    const z = 1.6 + ((pathIndex % 5) - 2) * 0.055;
    const points = [new THREE.Vector3(xAt(position), yAt(layer), z)];
    for (const step of path) {
      if (step === "R") position += 1;
      if (step === "U") layer += 1;
      points.push(new THREE.Vector3(xAt(position), yAt(layer), z));
    }
    const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.02);
    const line = tubeFromCurve(curve, COLORS.orange, pathIndex < 6 ? 0.026 : 0.014, pathIndex < 6 ? 0.35 : 0.1, 84);
    line.userData.kind = "path";
    pathGroup.add(line);
    if (pathIndex < 6) {
      addParticle(curve, COLORS.orange, 0.075, pathIndex / 6, 0.065, ["paths"]);
    }
  });

  const origin = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.28, 0),
    physicalMaterial(COLORS.orange, { emissiveIntensity: 2.2 }),
  );
  origin.position.set(xAt(0), yAt(0), 1.6);
  const target = origin.clone();
  target.position.set(xAt(3), yAt(3), 1.6);
  pathGroup.add(origin, target);
  pathGroup.visible = false;
}

function createBackdrop() {
  const grid = new THREE.GridHelper(24, 24, 0x34303f, 0x1d1c25);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -1.7;
  grid.material.transparent = true;
  grid.material.opacity = 0.23;
  scene.add(grid);

  const starsGeometry = new THREE.BufferGeometry();
  const stars = [];
  for (let i = 0; i < 260; i += 1) {
    stars.push((Math.random() - 0.5) * 30, (Math.random() - 0.5) * 22, -2.4 - Math.random() * 5);
  }
  starsGeometry.setAttribute("position", new THREE.Float32BufferAttribute(stars, 3));
  const starField = new THREE.Points(starsGeometry, new THREE.PointsMaterial({ color: 0x8b819d, size: 0.022, transparent: true, opacity: 0.48 }));
  scene.add(starField);
}

createResidualColumns();
for (let layer = 0; layer < LAYERS; layer += 1) {
  for (let position = 0; position < TOKENS.length; position += 1) createModule(layer, position);
}
createKVConnections();
createLayerLabels();
createCausalPaths();
createBackdrop();

let mode = "overview";
let isPlaying = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let speed = 1;
let hovered = null;
let transitioning = false;
let cameraGoal = camera.position.clone();
let targetGoal = controls.target.clone();

const modeCopy = {
  overview: {
    index: "01 / 03",
    caption: "The two-highway system",
    hint: "Drag to orbit · scroll to zoom · select any component",
    id: "FLOW / 01",
    icon: "dual",
    title: "The architecture is a <em>causal lattice.</em>",
    description: "Every point can inherit state from the layer below and retrieve layer-matched values from positions to its left. Together these axes create many distinct computational histories.",
  },
  block: {
    index: "02 / 03",
    caption: "One layer, one position",
    hint: "The current state queries a causal K/V cache",
    id: "BLOCK / L2·T2",
    icon: "node",
    iconColor: "#35d6ff",
    title: "One block performs <em>two residual updates.</em>",
    description: "First, causal attention reads layer-matched keys and values and adds its result. Then the position-wise MLP transforms that updated state and adds a second result. Layer normalization precedes each sublayer.",
  },
  paths: {
    index: "03 / 03",
    caption: "Many histories, one destination",
    hint: "Orange traces interleave horizontal and vertical steps",
    id: "PATHS / 3×3",
    icon: "node",
    iconColor: "#ffae57",
    title: "Order creates <em>combinatorial histories.</em>",
    description: "To move 3 layers up and 3 positions right using unit causal steps, a signal can interleave those moves in twenty distinct orders. Larger displacements grow as a binomial coefficient.",
    stat: "C(3 + 3, 3) = 20",
  },
};

function setInspector(copy) {
  document.querySelector("#signalId").textContent = copy.id;
  document.querySelector(".inspector-topline .eyebrow").textContent = copy.kicker || "SELECTED SIGNAL";
  const icon = copy.icon === "dual"
    ? '<div class="signal-icon dual" aria-hidden="true"><span></span><span></span></div>'
    : `<div class="signal-icon node" style="color:${copy.iconColor || "#35d6ff"}" aria-hidden="true"></div>`;
  const stat = copy.stat ? `<div class="path-stat"><strong>${copy.stat}</strong><span>monotone unit-step paths</span></div>` : "";
  document.querySelector("#inspectorContent").innerHTML = `${icon}<h2>${copy.title}</h2><div><p>${copy.description}</p>${stat}</div>`;
}

function opacityFor(object, nextMode) {
  const kind = object.userData.kind;
  if (nextMode === "overview") {
    if (kind === "connection") return object.userData.focus ? 0.72 : object.userData.baseOpacity;
    return object.userData.baseOpacity ?? object.material?.opacity ?? 1;
  }
  if (nextMode === "block") {
    const isFocusModule = object.parent?.userData.layer === 1 && object.parent?.userData.position === 2;
    const isIncoming = kind === "connection" && object.userData.layer === 1 && object.userData.target === 2;
    const isFocusResidual = kind === "residual" && object.userData.position === 2;
    if (isFocusModule || isIncoming || isFocusResidual) return Math.max(object.userData.baseOpacity ?? 1, 0.74);
    if (kind === "connection") return 0.012;
    return 0.055;
  }
  if (nextMode === "paths") {
    if (kind === "connection") return 0.012;
    if (kind === "module") return 0.035;
    if (kind === "residual") return 0.09;
    return object.userData.baseOpacity ?? object.material?.opacity ?? 1;
  }
  return 1;
}

function applyModeOpacity(nextMode) {
  [...modules, ...connections, ...residuals].forEach((object) => {
    if (!object.material || object.material.opacity === undefined) return;
    const opacity = opacityFor(object, nextMode);
    object.material.opacity = opacity;
    object.material.depthWrite = opacity > 0.45;
  });
  labelGroup.children.forEach((label) => {
    label.material.opacity = nextMode === "block" ? 0.12 : nextMode === "paths" ? 0.18 : 0.72;
  });
}

function setMode(nextMode) {
  mode = nextMode;
  document.querySelectorAll(".mode-button").forEach((button) => {
    const active = button.dataset.mode === nextMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });

  const copy = modeCopy[nextMode];
  document.querySelector(".caption-index").textContent = copy.index;
  const caption = document.querySelector("#sceneCaption div");
  caption.querySelector("strong").textContent = copy.caption;
  caption.querySelector("p").textContent = copy.hint;
  setInspector(copy);
  pathGroup.visible = nextMode === "paths";
  applyModeOpacity(nextMode);

  if (nextMode === "overview") {
    cameraGoal.set(10.8, 5.3, 18.4);
    targetGoal.set(0, 0, 0);
  } else if (nextMode === "block") {
    cameraGoal.set(4.5, yAt(1) + 1.2, 7.2);
    targetGoal.set(xAt(2), yAt(1) + 0.25, 0.3);
  } else {
    cameraGoal.set(9.8, 5.8, 18.8);
    targetGoal.set(-1.1, 0, 1.2);
  }
  transitioning = true;
}

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    setMode(button.dataset.mode);
    const url = new URL(window.location.href);
    url.searchParams.set("view", button.dataset.mode);
    window.history.replaceState({}, "", url);
  });
});

const playButton = document.querySelector("#playButton");
function syncPlayButton() {
  playButton.classList.toggle("paused", !isPlaying);
  playButton.setAttribute("aria-label", isPlaying ? "Pause animation" : "Play animation");
  playButton.setAttribute("aria-pressed", String(isPlaying));
}
playButton.addEventListener("click", () => {
  isPlaying = !isPlaying;
  syncPlayButton();
});
syncPlayButton();

document.querySelector("#resetButton").addEventListener("click", () => setMode(mode));
document.querySelector("#speedRange").addEventListener("input", (event) => {
  speed = Number(event.target.value);
});

const helpDialog = document.querySelector("#helpDialog");
document.querySelector("#helpButton").addEventListener("click", () => helpDialog.showModal());
document.querySelector("#closeHelp").addEventListener("click", () => helpDialog.close());
helpDialog.addEventListener("click", (event) => {
  if (event.target === helpDialog) helpDialog.close();
});

const raycaster = new THREE.Raycaster();
raycaster.params.Line.threshold = 0.15;
const pointer = new THREE.Vector2();

function intersectionsAt(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(pickables, false);
}

renderer.domElement.addEventListener("pointermove", (event) => {
  const hit = intersectionsAt(event)[0]?.object ?? null;
  if (hovered !== hit) {
    if (hovered?.material?.emissiveIntensity !== undefined) hovered.material.emissiveIntensity /= 1.8;
    hovered = hit;
    if (hovered?.material?.emissiveIntensity !== undefined) hovered.material.emissiveIntensity *= 1.8;
    renderer.domElement.style.cursor = hit ? "pointer" : "grab";
  }
});

renderer.domElement.addEventListener("click", (event) => {
  const info = intersectionsAt(event)[0]?.object?.userData.info;
  if (!info) return;
  setInspector({
    id: info.id,
    kicker: info.kicker,
    icon: "node",
    iconColor: info.color,
    title: info.title,
    description: info.description,
  });
});

renderer.domElement.addEventListener("pointerdown", () => {
  transitioning = false;
});

const clock = new THREE.Clock();
let elapsed = 0;
function animate() {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), 0.05);
  if (isPlaying) elapsed += delta * speed;

  flowParticles.forEach((particle) => {
    const flow = particle.userData.flow;
    const visible = flow.modes.includes(mode);
    particle.visible = visible;
    if (!visible) return;
    const t = (flow.offset + elapsed * flow.speed) % 1;
    particle.position.copy(flow.curve.getPoint(t));
    const pulse = 0.92 + Math.sin((elapsed * 5 + flow.offset * 12)) * 0.14;
    particle.scale.setScalar(pulse);
  });

  modules.forEach((object, index) => {
    if (object.material?.emissiveIntensity === undefined || !isPlaying) return;
    const base = object.geometry?.type === "SphereGeometry" ? 1.55 : object.geometry?.type === "RoundedBoxGeometry" ? 0.58 : 0.32;
    object.material.emissiveIntensity = base + Math.sin(elapsed * 1.2 + index * 0.17) * 0.08;
  });

  if (transitioning) {
    camera.position.lerp(cameraGoal, 0.055);
    controls.target.lerp(targetGoal, 0.055);
    if (camera.position.distanceTo(cameraGoal) < 0.03 && controls.target.distanceTo(targetGoal) < 0.02) transitioning = false;
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

const resizeObserver = new ResizeObserver(() => {
  const width = mount.clientWidth;
  const height = mount.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});
resizeObserver.observe(mount);

const requestedMode = urlParameters.get("view");
setMode(["overview", "block", "paths"].includes(requestedMode) ? requestedMode : "overview");
