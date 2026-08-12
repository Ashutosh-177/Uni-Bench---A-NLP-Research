// Reusable 3D bar chart for the leaderboard. Each results panel gets its
// own instance/scene -- deliberately simple geometry (boxes + wireframe
// edges, no text sprites) with real interaction: drag to orbit, hover a
// bar for its exact value via raycasting. Labels live in the HTML legend
// beside the chart, not baked into the 3D scene, so they stay crisp at
// any zoom level.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SERIES_COLORS = [0x3987e5, 0xd95926, 0x199e70, 0xc98500, 0xd55181, 0x9085e9];

export function seriesColor(index) {
  return SERIES_COLORS[index % SERIES_COLORS.length];
}
export function seriesColorHex(index) {
  return "#" + seriesColor(index).toString(16).padStart(6, "0");
}

/**
 * @param {HTMLElement} container - the `.chart-3d-shell` element
 * @returns {{ update: (items: {label:string, value:number, colorIndex:number}[], opts?: {max?:number, valueSuffix?:string}) => void, dispose: () => void }}
 */
export function createBarChart3D(container) {
  const width = container.clientWidth || 600;
  const height = container.clientHeight || 420;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
  camera.position.set(5.2, 4.4, 6.6);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 4;
  controls.maxDistance = 16;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.target.set(0, 0.6, 0);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;
  controls.addEventListener("start", () => (controls.autoRotate = false));

  // lighting -- soft key + fill, no harsh shadows (this is a UI chart, not a render)
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(4, 8, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x3987e5, 0.25);
  fill.position.set(-5, 2, -3);
  scene.add(fill);

  // floor grid
  const grid = new THREE.GridHelper(10, 20, 0x2c2c2a, 0x1c1e21);
  grid.position.y = 0;
  scene.add(grid);

  const barGroup = new THREE.Group();
  scene.add(barGroup);

  // tooltip element
  const tooltip = document.createElement("div");
  tooltip.className = "chart-3d-tooltip";
  container.style.position = container.style.position || "relative";
  container.appendChild(tooltip);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hoverable = [];

  function onPointerMove(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(hoverable, false);
    if (hits.length) {
      const obj = hits[0].object;
      tooltip.textContent = `${obj.userData.label}  ${obj.userData.value}`;
      tooltip.style.left = `${e.clientX - rect.left}px`;
      tooltip.style.top = `${e.clientY - rect.top}px`;
      tooltip.classList.add("visible");
      document.body.style.cursor = "pointer";
    } else {
      tooltip.classList.remove("visible");
      document.body.style.cursor = "default";
    }
  }
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerleave", () => tooltip.classList.remove("visible"));

  const BAR_SIZE = 0.9;
  const GAP = 0.55;
  const MAX_HEIGHT = 3.2;

  function update(items, opts = {}) {
    barGroup.clear();
    hoverable = [];
    const max = opts.max ?? Math.max(1e-6, ...items.map(i => i.value));
    const totalWidth = items.length * (BAR_SIZE + GAP) - GAP;
    const startX = -totalWidth / 2 + BAR_SIZE / 2;

    items.forEach((item, i) => {
      const targetH = Math.max(0.02, (item.value / max) * MAX_HEIGHT);
      const color = seriesColor(item.colorIndex);

      const geo = new THREE.BoxGeometry(BAR_SIZE, targetH, BAR_SIZE);
      geo.translate(0, targetH / 2, 0);
      const mat = new THREE.MeshStandardMaterial({
        color, roughness: 0.45, metalness: 0.12, transparent: true, opacity: 0.92,
      });
      const bar = new THREE.Mesh(geo, mat);
      bar.position.set(startX + i * (BAR_SIZE + GAP), 0, 0);
      bar.scale.y = 0.001; // animated in below
      bar.userData = { label: item.label, value: opts.valueSuffix ? `${item.value}${opts.valueSuffix}` : item.value };

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 })
      );
      bar.add(edges);

      barGroup.add(bar);
      hoverable.push(bar);

      // stagger + ease the grow-in animation
      const delay = i * 90;
      const start = performance.now() + delay;
      const dur = 700;
      function grow(now) {
        const p = Math.min(1, Math.max(0, (now - start) / dur));
        const eased = 1 - Math.pow(1 - p, 3);
        bar.scale.y = Math.max(0.001, eased);
        if (p < 1) requestAnimationFrame(grow);
      }
      requestAnimationFrame(grow);
    });
  }

  function onResize() {
    const w = container.clientWidth || width;
    const h = container.clientHeight || height;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  const resizeObserver = new ResizeObserver(onResize);
  resizeObserver.observe(container);

  let disposed = false;
  function loop() {
    if (disposed) return;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  loop();

  function dispose() {
    disposed = true;
    resizeObserver.disconnect();
    renderer.domElement.removeEventListener("pointermove", onPointerMove);
    renderer.dispose();
  }

  return { update, dispose };
}
