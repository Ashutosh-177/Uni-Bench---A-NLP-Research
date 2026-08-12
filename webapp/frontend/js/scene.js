// Ambient 3D background: a quiet constellation of nodes and connections,
// representing the model pool being compared. Deliberately restrained --
// low opacity, slow drift, subtle mouse parallax -- so it reads as
// environment, not decoration, and never fights the data-dense foreground.
import * as THREE from "three";

const canvas = document.getElementById("bg-scene");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const NODE_COUNT = 64;
const CONNECT_DIST = 3.6;
const FIELD_RADIUS = 9;

const SERIES_COLORS = [0x3987e5, 0xd95926, 0x199e70, 0xc98500, 0xd55181, 0x9085e9];

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x08090a, 0.045);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0.6, 13);

const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);

// ---- node field ----
const positions = [];
for (let i = 0; i < NODE_COUNT; i++) {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const r = FIELD_RADIUS * (0.35 + 0.65 * Math.random());
  positions.push(new THREE.Vector3(
    r * Math.sin(phi) * Math.cos(theta),
    r * Math.sin(phi) * Math.sin(theta) * 0.6,
    r * Math.cos(phi)
  ));
}

const group = new THREE.Group();
scene.add(group);

// Nodes as instanced spheres, colored from the fixed categorical palette
// (cycled here only because this is ambient decoration, not a data
// encoding -- charts elsewhere assign these colors to specific models).
const nodeGeo = new THREE.IcosahedronGeometry(0.045, 1);
const nodeMats = SERIES_COLORS.map(c => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.55 }));
positions.forEach((pos, i) => {
  const mesh = new THREE.Mesh(nodeGeo, nodeMats[i % nodeMats.length]);
  mesh.position.copy(pos);
  group.add(mesh);
});

// Connections between nearby nodes -- a single LineSegments buffer, built
// once (static topology; the whole group rotates as a rigid body).
const linePositions = [];
for (let i = 0; i < positions.length; i++) {
  for (let j = i + 1; j < positions.length; j++) {
    if (positions[i].distanceTo(positions[j]) < CONNECT_DIST) {
      linePositions.push(positions[i].x, positions[i].y, positions[i].z);
      linePositions.push(positions[j].x, positions[j].y, positions[j].z);
    }
  }
}
const lineGeo = new THREE.BufferGeometry();
lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
const lineMat = new THREE.LineBasicMaterial({ color: 0x2c3540, transparent: true, opacity: 0.28 });
group.add(new THREE.LineSegments(lineGeo, lineMat));

// ---- pointer parallax ----
let targetRotX = 0, targetRotY = 0;
let curRotX = 0, curRotY = 0;
window.addEventListener("pointermove", (e) => {
  const nx = (e.clientX / window.innerWidth) * 2 - 1;
  const ny = (e.clientY / window.innerHeight) * 2 - 1;
  targetRotY = nx * 0.18;
  targetRotX = ny * 0.1;
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

let t = 0;
function animate() {
  requestAnimationFrame(animate);
  if (!reduceMotion) {
    t += 0.0018;
    group.rotation.y = t + curRotY;
    group.rotation.x = Math.sin(t * 0.6) * 0.05 + curRotX;
    curRotX += (targetRotX - curRotX) * 0.03;
    curRotY += (targetRotY - curRotY) * 0.03;
  }
  renderer.render(scene, camera);
}

if (reduceMotion) {
  renderer.render(scene, camera);
} else {
  animate();
}
