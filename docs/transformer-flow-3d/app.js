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
  output: 0xf2e7c9,
};

const TOKENS = ["You", "are", "absolutely", "correct"];
const LAYERS = 4;
const X_SPACING = 3.35;
const Y_SPACING = 2.65;
const BLOCK_FOCUS = { layer: 1, position: 2 };
const OVERVIEW_FOCUS = { layer: 2, target: TOKENS.length - 1 };
const UP = new THREE.Vector3(0, 1, 0);
const WHITE = new THREE.Color(0xffffff);
const compactDisplay = window.matchMedia("(max-width: 720px)").matches;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const xAt = (position) => (position - (TOKENS.length - 1) / 2) * X_SPACING;
const yAt = (layer) => (layer - (LAYERS - 1) / 2) * Y_SPACING;
const urlParameters = new URLSearchParams(window.location.search);

if (urlParameters.get("stage") === "1") document.body.classList.add("stage-only");

const mount = document.querySelector("#scene");
const renderer = new THREE.WebGLRenderer({
  antialias: !compactDisplay,
  alpha: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, compactDisplay ? 1.15 : 1.5));
renderer.setSize(mount.clientWidth, mount.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.setClearColor(COLORS.background, 0);
mount.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(COLORS.background, 0.028);

const camera = new THREE.PerspectiveCamera(35, mount.clientWidth / mount.clientHeight, 0.1, 100);
camera.position.set(9.6, 5.4, 18.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.enablePan = false;
controls.minDistance = 6;
controls.maxDistance = 30;
controls.maxPolarAngle = Math.PI * 0.82;
controls.target.set(0, 0, 0);

scene.add(new THREE.HemisphereLight(0xb8c8ff, 0x100b1f, 1.25));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.7);
keyLight.position.set(5, 9, 12);
scene.add(keyLight);
const violetLight = new THREE.PointLight(COLORS.violet, 15, 20, 2);
violetLight.position.set(-4, 1, 5);
scene.add(violetLight);
const cyanLight = new THREE.PointLight(COLORS.cyan, 8, 16, 2);
cyanLight.position.set(5, -1, 4);
scene.add(cyanLight);

const systemGroup = new THREE.Group();
const residualGroup = new THREE.Group();
const connectionGroup = new THREE.Group();
const moduleGroup = new THREE.Group();
const labelGroup = new THREE.Group();
const outputGroup = new THREE.Group();
const pathGroup = new THREE.Group();
systemGroup.add(residualGroup, connectionGroup, moduleGroup, labelGroup, outputGroup);
scene.add(systemGroup, pathGroup);

const pickables = [];
const residualParts = [];
const connections = [];
const moduleParts = [];
const flowPulses = [];

const GEOMETRY = {
  kv: new THREE.SphereGeometry(0.18, compactDisplay ? 12 : 18, compactDisplay ? 8 : 12),
  attention: new RoundedBoxGeometry(0.9, 0.56, 0.74, 4, 0.1),
  mlp: new RoundedBoxGeometry(1.02, 0.52, 0.7, 4, 0.1),
  add: new THREE.TorusGeometry(0.15, 0.028, 7, 18),
  norm: new THREE.TorusGeometry(0.11, 0.018, 6, 16),
  pulse: new THREE.ConeGeometry(0.067, 0.3, 7),
  pulseGlow: new THREE.SphereGeometry(0.15, 8, 6),
  output: new RoundedBoxGeometry(1.4, 0.42, 0.6, 4, 0.1),
};

function standardMaterial(color, options = {}) {
  const opacity = options.opacity ?? 0.7;
  const material = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: options.emissiveIntensity ?? 0.42,
    roughness: options.roughness ?? 0.34,
    metalness: options.metalness ?? 0.12,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  material.userData.baseEmissiveIntensity = material.emissiveIntensity;
  return material;
}

function basicMaterial(color, opacity = 1, additive = opacity < 0.45) {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
}

function registerPart(object, kind, options = {}) {
  object.userData.kind = kind;
  object.userData.baseOpacity = options.baseOpacity ?? object.material?.opacity ?? 1;
  object.userData.layer = options.layer;
  object.userData.position = options.position;
  if (object.material?.color) object.userData.baseColor = object.material.color.getHex();
  return object;
}

function tubeFromCurve(curve, color, radius, opacity, segments = 40) {
  const tubularSegments = Math.max(10, Math.round(segments * (compactDisplay ? 0.68 : 1)));
  const geometry = new THREE.TubeGeometry(curve, tubularSegments, radius, compactDisplay ? 4 : 6, false);
  const mesh = new THREE.Mesh(geometry, basicMaterial(color, opacity));
  mesh.userData.curve = curve;
  mesh.userData.baseOpacity = opacity;
  return mesh;
}

function lineCurve(a, b) {
  return new THREE.LineCurve3(a, b);
}

function addArrow(position, direction, color, scale = 1, opacity = 1) {
  const arrow = new THREE.Mesh(GEOMETRY.pulse, basicMaterial(color, opacity, opacity < 0.5));
  arrow.position.copy(position);
  arrow.scale.setScalar(scale);
  arrow.quaternion.setFromUnitVectors(UP, direction.clone().normalize());
  arrow.userData.baseOpacity = opacity;
  return arrow;
}

function createTextSprite(text, color = "#8a8892", fontSize = 34, scale = [3.55, 0.66]) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 512;
  canvas.height = 96;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = `500 ${fontSize}px DM Mono, monospace`;
  context.fillStyle = color;
  context.fillText(text, 12, 58);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: 0.72,
    depthWrite: false,
  }));
  sprite.scale.set(scale[0], scale[1], 1);
  sprite.userData.baseOpacity = 0.72;
  return sprite;
}

function addFlowPulse(curve, color, options = {}) {
  const pulse = new THREE.Mesh(GEOMETRY.pulse, basicMaterial(color, options.opacity ?? 0.95, true));
  const glow = new THREE.Mesh(GEOMETRY.pulseGlow, basicMaterial(color, 0.12, true));
  pulse.add(glow);
  pulse.scale.setScalar(options.scale ?? 1);
  pulse.userData.flow = {
    curve,
    offset: options.offset ?? 0,
    speed: options.speed ?? 0.12,
    modes: options.modes ?? ["overview"],
    sequence: options.sequence,
    scale: options.scale ?? 1,
  };
  scene.add(pulse);
  flowPulses.push(pulse);
  return pulse;
}

function tokenCoordinate(position) {
  const displacement = position - (TOKENS.length - 1);
  return displacement === 0 ? "t" : `t − ${Math.abs(displacement)}`;
}

function componentLocation(layer, position) {
  return `Layer ${layer + 1} · position ${tokenCoordinate(position)} · “${TOKENS[position]}”`;
}

function tagPickable(mesh, info) {
  mesh.userData.info = info;
  mesh.name = info.name;
  pickables.push(mesh);
  return mesh;
}

function makeInfo(name, location, kicker, title, description, color) {
  return { name, location, kicker, title, description, color };
}

function addModulePart(group, object, layer, position, info) {
  registerPart(object, "module", { layer, position });
  group.add(object);
  moduleParts.push(object);
  if (info) tagPickable(object, info);
  return object;
}

function createResidualColumns() {
  const yMin = yAt(0) - 1.35;
  const yMax = yAt(LAYERS - 1) + 1.46;

  for (let position = 0; position < TOKENS.length; position += 1) {
    const x = xAt(position);
    const curve = lineCurve(new THREE.Vector3(x, yMin, 0), new THREE.Vector3(x, yMax, 0));
    const halo = registerPart(tubeFromCurve(curve, COLORS.graphite, 0.105, 0.045, 60), "residual", { position });
    const core = registerPart(tubeFromCurve(curve, COLORS.graphite, 0.034, 0.52, 60), "residual", { position });
    halo.userData.role = "halo";
    core.userData.role = "core";
    residualGroup.add(halo, core);
    residualParts.push(halo, core);
    tagPickable(core, makeInfo(
      "Residual stream",
      `Position ${tokenCoordinate(position)} · “${TOKENS[position]}” · all shown layers`,
      "VERTICAL STATE HIGHWAY",
      "The residual stream carries the <em>working state upward.</em>",
      "Each sublayer reads a normalized copy of this hidden state and writes an update back by addition. The stream itself is not a lossless history; its fixed model width is the shared representational bottleneck.",
      "#d7d5dc",
    ));

    for (let layer = 0; layer < LAYERS; layer += 1) {
      const arrow = registerPart(
        addArrow(new THREE.Vector3(x, yAt(layer) + 1.18, 0), UP, COLORS.graphite, 0.66, 0.58),
        "residual",
        { position },
      );
      arrow.userData.role = "arrow";
      residualGroup.add(arrow);
      residualParts.push(arrow);
    }

    addFlowPulse(curve, COLORS.graphite, {
      offset: position * 0.13,
      speed: 0.105,
      scale: 0.76,
      modes: ["overview"],
    });
    addFlowPulse(curve, COLORS.graphite, {
      offset: 0.5 + position * 0.13,
      speed: 0.105,
      scale: 0.62,
      opacity: 0.72,
      modes: ["overview"],
    });
  }
}

function createModule(layer, position) {
  const x = xAt(position);
  const y = yAt(layer);
  const location = componentLocation(layer, position);
  const group = new THREE.Group();
  group.position.set(x, y, 0);
  group.userData.layer = layer;
  group.userData.position = position;

  const kvStemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, -0.9, 0.02),
    new THREE.Vector3(0, -0.68, 0.24),
    new THREE.Vector3(0, -0.55, 0.58),
  ]);

  const preAttentionNorm = new THREE.Mesh(GEOMETRY.norm, standardMaterial(0x9ba8bc, {
    opacity: 0.62,
    emissiveIntensity: 0.26,
    roughness: 0.45,
  }));
  preAttentionNorm.rotation.x = Math.PI / 2;
  preAttentionNorm.position.set(0, -0.88, 0.03);
  addModulePart(group, preAttentionNorm, layer, position, makeInfo(
    "Pre-attention RMSNorm",
    location,
    "z = RMSNorm(x)",
    "Attention projections read a <em>normalized state.</em>",
    "Modern decoders normalize inside the residual branch with RMSNorm: divide by the root-mean-square, rescale with learned gains — no mean subtraction, no bias. Q, K, and V are projected from z while the raw x continues along the bypass.",
    "#aab5c7",
  ));

  const kvStem = tubeFromCurve(kvStemCurve, COLORS.violet, 0.04, 0.68, 14);
  addModulePart(group, kvStem, layer, position, makeInfo(
    "K/V projection branch",
    location,
    "NORMALIZED RESIDUAL READ",
    "This branch projects state into <em>keys and values.</em>",
    "The normalized state is multiplied by W_K and W_V, and RoPE rotates the key by angles set by this position's index. The resulting pair is what queries at this and later positions read at this layer.",
    "#a78bfa",
  ));

  const kv = new THREE.Mesh(GEOMETRY.kv, standardMaterial(COLORS.violet, {
    opacity: 0.64,
    emissiveIntensity: 1.05,
    roughness: 0.25,
  }));
  kv.position.set(0, -0.51, 0.62);
  addModulePart(group, kv, layer, position, makeInfo(
    "Key/value projection",
    location,
    "K AND V VECTORS",
    "This position becomes an <em>addressable memory.</em>",
    "K encodes how a later query can match this position; V encodes what is delivered after the match. With grouped-query attention there are only a few K/V heads, each shared by a group of query heads — and during generation this pair is computed once and kept in the KV cache.",
    "#a78bfa",
  ));

  const localKVReadCurve = new THREE.CubicBezierCurve3(
    new THREE.Vector3(0, -0.51, 0.62),
    new THREE.Vector3(0.28, -0.34, 0.84),
    new THREE.Vector3(0.28, -0.08, 0.84),
    new THREE.Vector3(0.08, 0.02, 0.73),
  );
  const localKVRead = tubeFromCurve(localKVReadCurve, COLORS.violet, 0.024, 0.6, 16);
  addModulePart(group, localKVRead, layer, position, makeInfo(
    "Current-position K/V read",
    location,
    "SELF POSITION IS ALLOWED",
    "Causal attention includes the <em>current position.</em>",
    "The causal mask permits a query at position t to attend to keys and values at positions ≤t, including its own K/V pair. It only forbids positions to the right.",
    "#a78bfa",
  ));

  const queryStemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, -0.88, 0.03),
    new THREE.Vector3(-0.2, -0.4, 0.2),
    new THREE.Vector3(-0.12, -0.08, 0.48),
  ]);
  const queryStem = tubeFromCurve(queryStemCurve, COLORS.cyan, 0.024, 0.56, 14);
  addModulePart(group, queryStem, layer, position, makeInfo(
    "Query projection",
    location,
    "q = RoPE(RMSNorm(x)Wq)",
    "The current position forms a <em>query vector.</em>",
    "Within each head, q represents what this position is looking for. RoPE rotates q and each k by position-dependent angles, so the score q·k depends on the relative offset between tokens — no position vector is ever added to the residual stream.",
    "#35d6ff",
  ));

  const attention = new THREE.Mesh(GEOMETRY.attention, standardMaterial(COLORS.cyan, {
    opacity: 0.5,
    emissiveIntensity: 0.56,
    roughness: 0.22,
  }));
  attention.position.set(0, 0.02, 0.36);
  attention.renderOrder = 3;
  addModulePart(group, attention, layer, position, makeInfo(
    "Causal self-attention block",
    location,
    "QK MATCH · V RETRIEVAL",
    "A query turns context into a <em>weighted value read.</em>",
    `The query at “${TOKENS[position]}” scores keys from positions 0…${position}. Softmax gives every query head its own weight distribution; in grouped-query attention several query heads read the same shared K/V head. The weighted values are concatenated and output-projected into a residual update.`,
    "#35d6ff",
  ));

  const attnOutputCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.29, 0.36),
    new THREE.Vector3(0, 0.42, 0.25),
    new THREE.Vector3(0, 0.5, 0.04),
  ]);
  const attnOutput = tubeFromCurve(attnOutputCurve, COLORS.cyan, 0.036, 0.7, 12);
  addModulePart(group, attnOutput, layer, position, makeInfo(
    "Attention output projection",
    location,
    "WRITE TO RESIDUAL STREAM",
    "The retrieved values become a <em>residual update.</em>",
    "Each head’s weighted value sum is concatenated with the others and multiplied by the output projection Wₒ before being added to the incoming residual state.",
    "#35d6ff",
  ));

  const addOne = new THREE.Mesh(GEOMETRY.add, standardMaterial(COLORS.graphite, {
    opacity: 0.72,
    emissiveIntensity: 0.18,
    roughness: 0.4,
  }));
  addOne.rotation.x = Math.PI / 2;
  addOne.position.set(0, 0.5, 0.02);
  addModulePart(group, addOne, layer, position, makeInfo(
    "First residual addition",
    location,
    "x + Attn(RMSNorm(x))",
    "Attention is <em>added, not substituted.</em>",
    "The incoming residual stream bypasses attention. Its state and the attention output are summed here to form the post-attention state u.",
    "#d7d5dc",
  ));

  const mlpInputCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.51, 0.03),
    new THREE.Vector3(0, 0.64, 0.14),
    new THREE.Vector3(0, 0.68, 0.3),
  ]);

  const preMlpNorm = new THREE.Mesh(GEOMETRY.norm, standardMaterial(0x9ba8bc, {
    opacity: 0.62,
    emissiveIntensity: 0.26,
    roughness: 0.45,
  }));
  preMlpNorm.rotation.x = Math.PI / 2;
  preMlpNorm.position.set(0, 0.6, 0.08);
  addModulePart(group, preMlpNorm, layer, position, makeInfo(
    "Pre-MLP normalization",
    location,
    "RMSNorm(u) READ",
    "The MLP reads a <em>normalized post-attention state.</em>",
    "After the first residual addition forms u, a second RMSNorm produces the MLP input. The unnormalized u remains on the residual bypass and is added back after the MLP.",
    "#aab5c7",
  ));

  const mlpInput = tubeFromCurve(mlpInputCurve, COLORS.graphite, 0.03, 0.58, 10);
  addModulePart(group, mlpInput, layer, position, makeInfo(
    "Normalized MLP input",
    location,
    "RMSNorm(u) READ",
    "The MLP reads the <em>post-attention state.</em>",
    "In this pre-norm block, the state after the first residual addition is normalized and fed into the MLP. The residual stream itself continues along the bypass.",
    "#d7d5dc",
  ));

  const mlp = new THREE.Mesh(GEOMETRY.mlp, standardMaterial(COLORS.magenta, {
    opacity: 0.48,
    emissiveIntensity: 0.52,
    roughness: 0.24,
  }));
  mlp.position.set(0, 0.88, 0.3);
  mlp.renderOrder = 3;
  addModulePart(group, mlp, layer, position, makeInfo(
    "Gated MLP (SwiGLU)",
    location,
    "PER-POSITION SwiGLU",
    "The MLP transforms the <em>current position locally.</em>",
    "The MLP mixes features within this one position only — no cross-token communication. Modern models use a gated form: an up- and a gate-projection, SiLU on the gate, an elementwise product, then a down-projection. In mixture-of-experts models a router activates a few expert MLPs per token here instead.",
    "#f34bb5",
  ));

  const topStemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 1.12, 0.3),
    new THREE.Vector3(0, 1.2, 0.12),
    new THREE.Vector3(0, 1.28, 0),
  ]);
  const topStem = tubeFromCurve(topStemCurve, COLORS.magenta, 0.036, 0.66, 11);
  addModulePart(group, topStem, layer, position, makeInfo(
    "MLP output update",
    location,
    "WRITE TO RESIDUAL STREAM",
    "The MLP writes a second <em>residual update.</em>",
    "The MLP output returns to the residual stream. This update is summed with u, the state that already contains the attention update.",
    "#f34bb5",
  ));

  const addTwo = new THREE.Mesh(GEOMETRY.add, standardMaterial(COLORS.graphite, {
    opacity: 0.72,
    emissiveIntensity: 0.18,
    roughness: 0.4,
  }));
  addTwo.rotation.x = Math.PI / 2;
  addTwo.position.set(0, 1.28, 0.01);
  addModulePart(group, addTwo, layer, position, makeInfo(
    "Second residual addition",
    location,
    "u + MLP(RMSNorm(u))",
    "The completed block state continues <em>upward.</em>",
    "The MLP update is added to the post-attention state u. The resulting state x at the next layer retains both bypass paths and both learned updates.",
    "#d7d5dc",
  ));

  moduleGroup.add(group);
}

function createKVConnections() {
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const y = yAt(layer);
    for (let target = 0; target < TOKENS.length; target += 1) {
      for (let source = 0; source < target; source += 1) {
        const distance = target - source;
        const start = new THREE.Vector3(xAt(source), y - 0.51, 0.62);
        const end = new THREE.Vector3(xAt(target) - 0.06, y + 0.02, 0.75);
        const zLane = 0.9 + distance * 0.25;
        const sag = 0.25 + distance * 0.055;
        const curve = new THREE.CubicBezierCurve3(
          start,
          new THREE.Vector3(xAt(source) + distance * 0.76, y - sag, zLane),
          new THREE.Vector3(xAt(target) - distance * 0.7, y - sag, zLane),
          end,
        );
        const isOverviewFocus = target === OVERVIEW_FOCUS.target && layer === OVERVIEW_FOCUS.layer;
        const isBlockInput = target === BLOCK_FOCUS.position && layer === BLOCK_FOCUS.layer;
        const baseOpacity = isOverviewFocus ? 0.56 : 0.085;
        const connection = registerPart(
          tubeFromCurve(curve, COLORS.violet, isOverviewFocus ? 0.032 : 0.022, baseOpacity, 28 + distance * 6),
          "connection",
          { baseOpacity },
        );
        connection.userData.layer = layer;
        connection.userData.source = source;
        connection.userData.target = target;
        connection.userData.focus = isOverviewFocus;
        connectionGroup.add(connection);
        connections.push(connection);
        tagPickable(connection, makeInfo(
          "Causal K/V retrieval edge",
          `Layer ${layer + 1} · “${TOKENS[source]}” → “${TOKENS[target]}”`,
          "PAST-TO-PRESENT ATTENTION EDGE",
          "A later query can <em>retrieve this value.</em>",
          `The query at “${TOKENS[target]}” scores the key stored at “${TOKENS[source]}”. The resulting per-head attention weight controls how much of that source value contributes to the weighted sum.`,
          "#a78bfa",
        ));

        if (isOverviewFocus) {
          addFlowPulse(curve, COLORS.violetBright, {
            offset: source * 0.17,
            speed: 0.16 + distance * 0.012,
            scale: 0.72,
            modes: ["overview"],
          });
          addFlowPulse(curve, COLORS.violetBright, {
            offset: 0.48 + source * 0.17,
            speed: 0.16 + distance * 0.012,
            scale: 0.54,
            opacity: 0.68,
            modes: ["overview"],
          });
        }

        if (isBlockInput) {
          addFlowPulse(curve, COLORS.violetBright, {
            scale: 0.76,
            modes: ["block"],
            sequence: { start: 0.2 + source * 0.025, duration: 0.27 },
          });
        }
      }
    }
  }
}

function createBlockSequence() {
  const x = xAt(BLOCK_FOCUS.position);
  const y = yAt(BLOCK_FOCUS.layer);
  const curve = (points) => new THREE.CatmullRomCurve3(points.map(([dx, dy, dz]) => new THREE.Vector3(x + dx, y + dy, dz)));

  addFlowPulse(curve([[0, -1.24, 0], [0, -0.95, 0], [0, -0.7, 0.08]]), COLORS.graphite, {
    scale: 0.78,
    modes: ["block"],
    sequence: { start: 0, duration: 0.15 },
  });
  addFlowPulse(curve([[0, -0.9, 0.02], [0, -0.68, 0.24], [0, -0.51, 0.62]]), COLORS.violetBright, {
    scale: 0.76,
    modes: ["block"],
    sequence: { start: 0.05, duration: 0.15 },
  });
  addFlowPulse(curve([[0, -0.78, 0.04], [0.17, -0.34, 0.25], [0.05, 0.02, 0.72]]), COLORS.cyan, {
    scale: 0.78,
    modes: ["block"],
    sequence: { start: 0.05, duration: 0.15 },
  });
  addFlowPulse(curve([[0, -0.51, 0.62], [0.28, -0.34, 0.84], [0.08, 0.02, 0.73]]), COLORS.violetBright, {
    scale: 0.72,
    modes: ["block"],
    sequence: { start: 0.2, duration: 0.27 },
  });
  addFlowPulse(curve([[0, 0.28, 0.36], [0, 0.42, 0.25], [0, 0.5, 0.04]]), COLORS.cyan, {
    scale: 0.72,
    modes: ["block"],
    sequence: { start: 0.49, duration: 0.11 },
  });
  addFlowPulse(curve([[0, 0.5, 0.03], [0, 0.64, 0.14], [0, 0.69, 0.3]]), COLORS.graphite, {
    scale: 0.66,
    modes: ["block"],
    sequence: { start: 0.6, duration: 0.1 },
  });
  addFlowPulse(curve([[0, 0.78, 0.3], [0, 0.9, 0.31], [0, 1.11, 0.3]]), COLORS.magenta, {
    scale: 0.76,
    modes: ["block"],
    sequence: { start: 0.69, duration: 0.15 },
  });
  addFlowPulse(curve([[0, 1.11, 0.3], [0, 1.2, 0.12], [0, 1.29, 0]]), COLORS.magenta, {
    scale: 0.72,
    modes: ["block"],
    sequence: { start: 0.82, duration: 0.1 },
  });
  addFlowPulse(curve([[0, 1.28, 0], [0, 1.38, 0], [0, 1.54, 0]]), COLORS.graphite, {
    scale: 0.76,
    modes: ["block"],
    sequence: { start: 0.91, duration: 0.09 },
  });
}

function createLayerLabels() {
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const sprite = createTextSprite(`LAYER  ${String(layer + 1).padStart(2, "0")}`);
    sprite.position.set(xAt(0) - 1.5, yAt(layer), -0.2);
    labelGroup.add(sprite);
  }
}

function createOutputHead() {
  const position = TOKENS.length - 1;
  const x = xAt(position);
  const y = yAt(LAYERS - 1) + 1.68;
  const output = new THREE.Mesh(GEOMETRY.output, standardMaterial(COLORS.output, {
    opacity: 0.28,
    emissiveIntensity: 0.5,
    roughness: 0.35,
  }));
  output.position.set(x, y, 0.22);
  registerPart(output, "module", { layer: LAYERS, position });
  moduleParts.push(output);
  outputGroup.add(output);
  tagPickable(output, makeInfo(
    "Final RMSNorm and language-model head",
    `Final state at position t · “${TOKENS[position]}”`,
    "RESIDUAL STATE → VOCABULARY LOGITS",
    "The last input position produces the <em>next-token distribution.</em>",
    "A final RMSNorm and the unembedding matrix map the last residual state to one logit per vocabulary token; softmax and a sampling rule pick the next token. During training the head is applied at every position in parallel; generation reads only the last position shown here.",
    "#f2e7c9",
  ));

  const stemCurve = lineCurve(new THREE.Vector3(x, yAt(LAYERS - 1) + 1.42, 0), new THREE.Vector3(x, y - 0.22, 0.12));
  const stem = registerPart(tubeFromCurve(stemCurve, COLORS.output, 0.03, 0.46, 10), "module", { layer: LAYERS, position });
  outputGroup.add(stem);
  moduleParts.push(stem);
  tagPickable(stem, output.userData.info);

  const label = createTextSprite("LM HEAD · LOGITS", "#bdb59f", 24, [2.8, 0.52]);
  label.position.set(x - 0.1, y + 0.34, 0.1);
  outputGroup.add(label);

  addFlowPulse(stemCurve, COLORS.output, {
    offset: 0.1,
    speed: 0.19,
    scale: 0.68,
    modes: ["overview"],
  });
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
    const z = 1.58 + ((pathIndex % 5) - 2) * 0.052;
    const points = [new THREE.Vector3(xAt(position), yAt(layer), z)];
    for (const step of path) {
      if (step === "R") position += 1;
      if (step === "U") layer += 1;
      points.push(new THREE.Vector3(xAt(position), yAt(layer), z));
    }
    const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.02);
    const prominent = pathIndex < 6;
    const line = tubeFromCurve(curve, COLORS.orange, prominent ? 0.025 : 0.012, prominent ? 0.32 : 0.075, 52);
    line.userData.kind = "path";
    pathGroup.add(line);
    tagPickable(line, makeInfo(
      "One unit-step computational history",
      `Path ${pathIndex + 1} of 20 · 3 right steps · 3 upward steps`,
      "PEDAGOGICAL CAUSAL SUBGRAPH",
      "This trace is one <em>ordering of six unit steps.</em>",
      "The twenty orange traces enumerate a nearest-neighbor abstraction only. Real self-attention can jump across several earlier positions in a single layer, so this binomial count is not the total number of paths in a production transformer.",
      "#ffae57",
    ));
    if (prominent) {
      addFlowPulse(curve, COLORS.orange, {
        offset: pathIndex / 6,
        speed: 0.075,
        scale: 0.76,
        modes: ["paths"],
      });
    }
  });

  const origin = new THREE.Mesh(new THREE.OctahedronGeometry(0.27, 0), standardMaterial(COLORS.orange, {
    opacity: 0.82,
    emissiveIntensity: 1.6,
  }));
  origin.position.set(xAt(0), yAt(0), 1.58);
  tagPickable(origin, makeInfo(
    "Path origin",
    "Layer 1 · position t − 3 · “You”",
    "SOURCE STATE",
    "The path family starts from <em>one earlier state.</em>",
    "Orange traces show different unit-step orders by which an influence can move upward through depth and rightward through causal attention.",
    "#ffae57",
  ));

  const target = new THREE.Mesh(origin.geometry, origin.material.clone());
  target.position.set(xAt(3), yAt(3), 1.58);
  tagPickable(target, makeInfo(
    "Path destination",
    "Layer 4 · position t · “correct”",
    "DESTINATION STATE",
    "Distinct histories recombine at <em>one later state.</em>",
    "The destination receives transformed contributions that followed many causal routes. Their effects are summed in the shared residual representation.",
    "#ffae57",
  ));
  pathGroup.add(origin, target);
  pathGroup.visible = false;
}

function createBackdrop() {
  const grid = new THREE.GridHelper(22, 22, 0x34303f, 0x1d1c25);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -1.7;
  grid.material.transparent = true;
  grid.material.opacity = 0.2;
  scene.add(grid);

  const starCount = compactDisplay ? 110 : 190;
  const starPositions = new Float32Array(starCount * 3);
  for (let index = 0; index < starCount; index += 1) {
    starPositions[index * 3] = (Math.random() - 0.5) * 28;
    starPositions[index * 3 + 1] = (Math.random() - 0.5) * 22;
    starPositions[index * 3 + 2] = -2.4 - Math.random() * 5;
  }
  const starsGeometry = new THREE.BufferGeometry();
  starsGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  scene.add(new THREE.Points(starsGeometry, new THREE.PointsMaterial({
    color: 0x8b819d,
    size: 0.022,
    transparent: true,
    opacity: 0.42,
  })));
}

createResidualColumns();
for (let layer = 0; layer < LAYERS; layer += 1) {
  for (let position = 0; position < TOKENS.length; position += 1) createModule(layer, position);
}
createKVConnections();
createBlockSequence();
createLayerLabels();
createOutputHead();
createCausalPaths();
createBackdrop();

let mode = "overview";
let isPlaying = !reducedMotion;
let speed = 1;
let hovered = null;
let selected = null;
let transitioning = false;
let cameraGoal = camera.position.clone();
let targetGoal = controls.target.clone();
let sceneVisible = true;
let animationFrame = 0;
let elapsed = 0;
let lastTimestamp = 0;

const modeCopy = {
  overview: {
    index: "01 / 03",
    caption: "The two-highway system",
    hint: "Drag to orbit · scroll to zoom · click any component",
    id: "FLOW / 01",
    icon: "dual",
    name: "Causal transformer lattice",
    location: "4 input positions · 4 representative layers",
    title: "The architecture is a <em>causal lattice.</em>",
    description: "Residual state moves through depth at each position. At each layer, attention directly retrieves keys and values from allowed source positions; the purple arcs are causal read edges, not a serial bus that copies one cache through the next.",
  },
  block: {
    index: "02 / 03",
    caption: "One layer, one position",
    hint: "Watch Q/K/V projection → causal retrieval → attention → MLP",
    id: "BLOCK / L2·T−1",
    icon: "node",
    iconColor: "#35d6ff",
    name: "One pre-norm decoder block",
    location: "Layer 2 · position t − 1 · “absolutely”",
    title: "One block performs <em>two residual updates.</em>",
    description: "Q, K, and V are projected from the normalized incoming state. Attention is added first; the MLP then reads the normalized post-attention state and its output is added second. The animated sequence makes both bypasses explicit.",
  },
  paths: {
    index: "03 / 03",
    caption: "Many histories, one destination",
    hint: "Orange traces enumerate a unit-step causal subgraph",
    id: "PATHS / 3×3",
    icon: "node",
    iconColor: "#ffae57",
    name: "Nearest-neighbor path abstraction",
    location: "3 layer steps · 3 position steps",
    title: "Unit steps create <em>combinatorial histories.</em>",
    description: "There are twenty orders for interleaving three upward and three rightward unit steps. This is a pedagogical subgraph: real attention includes long-range horizontal edges, so C(6,3) is not the total number of actual transformer paths.",
    stat: "C(3 + 3, 3) = 20",
  },
};

function setInspector(copy) {
  document.querySelector("#signalId").textContent = copy.id || copy.name.toUpperCase();
  document.querySelector(".inspector-topline .eyebrow").textContent = copy.kicker || "SELECTED COMPONENT";
  const icon = copy.icon === "dual"
    ? '<div class="signal-icon dual" aria-hidden="true"><span></span><span></span></div>'
    : `<div class="signal-icon node" style="color:${copy.iconColor || copy.color || "#35d6ff"}" aria-hidden="true"></div>`;
  const identity = `<div class="object-identity"><span>WHAT THIS IS</span><strong>${copy.name}</strong><small>${copy.location}</small></div>`;
  const stat = copy.stat ? `<div class="path-stat"><strong>${copy.stat}</strong><span>monotone nearest-neighbor paths</span></div>` : "";
  document.querySelector("#inspectorContent").innerHTML = `${icon}${identity}<h2>${copy.title}</h2><div><p>${copy.description}</p>${stat}</div>`;
}

function setSelectionPill(copy, visible) {
  const pill = document.querySelector("#selectionPill");
  pill.classList.toggle("visible", visible);
  if (!visible) return;
  pill.querySelector("i").style.background = copy.color || "#a78bfa";
  pill.querySelector("i").style.boxShadow = `0 0 10px ${copy.color || "#a78bfa"}`;
  pill.querySelector("strong").textContent = copy.name;
  pill.querySelector("small").textContent = copy.location;
}

function refreshEmphasis(object) {
  if (!object?.material) return;
  const amount = object === selected ? 0.42 : object === hovered ? 0.2 : 0;
  if (object.material.color && object.userData.baseColor !== undefined) {
    object.material.color.setHex(object.userData.baseColor).lerp(WHITE, amount);
  }
  if (object.material.emissiveIntensity !== undefined) {
    const base = object.material.userData.baseEmissiveIntensity ?? 0;
    object.material.emissiveIntensity = base + (object === selected ? 0.75 : object === hovered ? 0.38 : 0);
  }
}

function clearSelection() {
  const previous = selected;
  selected = null;
  refreshEmphasis(previous);
  setSelectionPill({}, false);
}

function selectObject(object) {
  const previous = selected;
  selected = object;
  refreshEmphasis(previous);
  refreshEmphasis(selected);
  const info = object.userData.info;
  setInspector({
    ...info,
    id: object.name.toUpperCase(),
    icon: "node",
    iconColor: info.color,
  });
  setSelectionPill(info, true);
  queueFrame();
}

function opacityFor(object, nextMode) {
  const kind = object.userData.kind;
  const base = object.userData.baseOpacity ?? object.material?.opacity ?? 1;
  if (nextMode === "overview") {
    if (kind === "connection") return object.userData.focus ? 0.56 : base;
    return base;
  }
  if (nextMode === "block") {
    const isFocusModule = object.userData.layer === BLOCK_FOCUS.layer && object.userData.position === BLOCK_FOCUS.position;
    const isIncoming = kind === "connection" && object.userData.layer === BLOCK_FOCUS.layer && object.userData.target === BLOCK_FOCUS.position;
    const isFocusResidual = kind === "residual" && object.userData.position === BLOCK_FOCUS.position;
    if (isFocusResidual) {
      if (object.userData.role === "halo") return 0.045;
      if (object.userData.role === "core") return 0.36;
      return 0.58;
    }
    if (isFocusModule || isIncoming) return Math.max(base, 0.58);
    if (kind === "connection") return 0.008;
    return 0.035;
  }
  if (nextMode === "paths") {
    if (kind === "connection") return 0.006;
    if (kind === "module") return 0.025;
    if (kind === "residual") return object.userData.role === "halo" ? 0.018 : 0.075;
  }
  return base;
}

function applyModeOpacity(nextMode) {
  [...moduleParts, ...connections, ...residualParts].forEach((object) => {
    if (!object.material || object.material.opacity === undefined) return;
    object.material.opacity = opacityFor(object, nextMode);
  });
  labelGroup.children.forEach((label) => {
    label.material.opacity = nextMode === "block" ? 0.1 : nextMode === "paths" ? 0.16 : 0.65;
  });
  outputGroup.children.forEach((object) => {
    if (object.isSprite) object.material.opacity = nextMode === "overview" ? 0.72 : 0.08;
  });
}

function setMode(nextMode) {
  clearSelection();
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
    cameraGoal.set(9.6, 5.4, 18.8);
    targetGoal.set(0, 0.15, 0);
  } else if (nextMode === "block") {
    cameraGoal.set(4.5, yAt(BLOCK_FOCUS.layer) + 1.15, 7.2);
    targetGoal.set(xAt(BLOCK_FOCUS.position), yAt(BLOCK_FOCUS.layer) + 0.25, 0.3);
  } else {
    cameraGoal.set(9.1, 5.7, 17.7);
    targetGoal.set(0, 0, 1.15);
  }
  transitioning = true;
  queueFrame();
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
  queueFrame();
});
syncPlayButton();

document.querySelector("#resetButton").addEventListener("click", () => setMode(mode));
document.querySelector("#speedRange").addEventListener("input", (event) => {
  speed = Number(event.target.value);
  queueFrame();
});

const helpDialog = document.querySelector("#helpDialog");
document.querySelector("#helpButton").addEventListener("click", () => helpDialog.showModal());
document.querySelector("#closeHelp").addEventListener("click", () => helpDialog.close());
helpDialog.addEventListener("click", (event) => {
  if (event.target === helpDialog) helpDialog.close();
});

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pointerStart = { x: 0, y: 0 };
let latestPointerEvent = null;
let pointerFrame = 0;

function intersectionsAt(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(pickables, false);
}

function updateHover(event) {
  const hit = intersectionsAt(event)[0]?.object ?? null;
  if (hovered === hit) return;
  const previous = hovered;
  hovered = hit;
  refreshEmphasis(previous);
  refreshEmphasis(hovered);
  renderer.domElement.style.cursor = hit ? "pointer" : "grab";
  queueFrame();
}

renderer.domElement.addEventListener("pointermove", (event) => {
  latestPointerEvent = event;
  if (pointerFrame) return;
  pointerFrame = window.requestAnimationFrame(() => {
    pointerFrame = 0;
    updateHover(latestPointerEvent);
  });
});

renderer.domElement.addEventListener("pointerdown", (event) => {
  pointerStart = { x: event.clientX, y: event.clientY };
  transitioning = false;
});

renderer.domElement.addEventListener("click", (event) => {
  if (Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5) return;
  const object = intersectionsAt(event)[0]?.object;
  if (object?.userData.info) selectObject(object);
});

function updateFlowPulses() {
  const sequenceCycle = (elapsed * 0.19) % 1;
  flowPulses.forEach((pulse) => {
    const flow = pulse.userData.flow;
    let visible = flow.modes.includes(mode);
    let progress = 0;

    if (visible && flow.sequence) {
      const local = (sequenceCycle - flow.sequence.start + 1) % 1;
      visible = local <= flow.sequence.duration;
      progress = visible ? local / flow.sequence.duration : 0;
    } else if (visible) {
      progress = (flow.offset + elapsed * flow.speed) % 1;
    }

    pulse.visible = visible;
    if (!visible) return;
    pulse.position.copy(flow.curve.getPoint(progress));
    pulse.quaternion.setFromUnitVectors(UP, flow.curve.getTangent(progress).normalize());
    const pulseScale = flow.scale * (0.9 + Math.sin(elapsed * 6 + flow.offset * 11) * 0.1);
    pulse.scale.setScalar(pulseScale);
  });
}

function queueFrame() {
  if (animationFrame || !sceneVisible || document.hidden) return;
  animationFrame = window.requestAnimationFrame(animate);
}

function animate(timestamp) {
  animationFrame = 0;
  const frameInterval = compactDisplay ? 1000 / 30 : 1000 / 60;
  if (lastTimestamp && timestamp - lastTimestamp < frameInterval - 1) {
    queueFrame();
    return;
  }
  const delta = lastTimestamp ? Math.min((timestamp - lastTimestamp) / 1000, 0.05) : 0;
  lastTimestamp = timestamp;
  if (isPlaying) elapsed += delta * speed;
  updateFlowPulses();

  if (transitioning) {
    camera.position.lerp(cameraGoal, 0.07);
    controls.target.lerp(targetGoal, 0.07);
    if (camera.position.distanceTo(cameraGoal) < 0.025 && controls.target.distanceTo(targetGoal) < 0.018) transitioning = false;
  }

  controls.update();
  renderer.render(scene, camera);
  if (isPlaying || transitioning) queueFrame();
}

controls.addEventListener("change", queueFrame);

const visibilityObserver = new IntersectionObserver(([entry]) => {
  sceneVisible = entry.isIntersecting;
  if (sceneVisible) {
    lastTimestamp = 0;
    queueFrame();
  }
}, { threshold: 0.01 });
visibilityObserver.observe(mount);

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    lastTimestamp = 0;
    queueFrame();
  }
});

const resizeObserver = new ResizeObserver(() => {
  const width = mount.clientWidth;
  const height = mount.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  queueFrame();
});
resizeObserver.observe(mount);

const requestedMode = urlParameters.get("view");
setMode(["overview", "block", "paths"].includes(requestedMode) ? requestedMode : "overview");
