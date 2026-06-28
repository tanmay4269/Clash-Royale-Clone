const state = {
  opponents: null,
  selectedOpponent: null,
  running: false,
  frameTimer: null,
};

const setup = document.getElementById("setup");
const game = document.getElementById("game");
const opponentSelect = document.getElementById("opponentSelect");
const opponentMeta = document.getElementById("opponentMeta");
const dependencyStatus = document.getElementById("dependencyStatus");
const startButton = document.getElementById("startButton");
const newMatchButton = document.getElementById("newMatchButton");
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const matchTitle = document.getElementById("matchTitle");
const matchMeta = document.getElementById("matchMeta");

function allOpponentOptions() {
  if (!state.opponents) return [];
  return [
    ...state.opponents.builtIns,
    ...state.opponents.bestByRun,
    ...state.opponents.checkpoints,
  ];
}

function optionById(id) {
  return allOpponentOptions().find((item) => item.id === id);
}

function addOptions(groupLabel, items) {
  if (!items.length) return;
  const group = document.createElement("optgroup");
  group.label = groupLabel;
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    group.appendChild(option);
  }
  opponentSelect.appendChild(group);
}

function renderOpponentSelect() {
  opponentSelect.innerHTML = "";
  addOptions("Built in", state.opponents.builtIns);
  addOptions("Best checkpoint per run", state.opponents.bestByRun);
  addOptions("All run 30 and 31 checkpoints", state.opponents.checkpoints);

  const firstBest = state.opponents.bestByRun[0];
  opponentSelect.value = firstBest ? firstBest.id : "random";
  updateOpponentMeta();
}

function updateOpponentMeta() {
  const selected = optionById(opponentSelect.value);
  state.selectedOpponent = selected;
  if (!selected) {
    opponentMeta.textContent = "";
    return;
  }

  const parts = [selected.architectureLabel];
  if (selected.runId) parts.push(selected.runId);
  if (selected.elo != null) parts.push(`Elo ${selected.elo}`);
  if (selected.checkpointIndex != null) parts.push(`checkpoint ${selected.checkpointIndex}`);
  opponentMeta.textContent = parts.join(", ");
}

async function loadOpponents() {
  const response = await fetch("/api/opponents");
  state.opponents = await response.json();
  renderOpponentSelect();
}

async function startMatch() {
  startButton.disabled = true;
  dependencyStatus.textContent = "";
  const response = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ opponentId: opponentSelect.value }),
  });
  const payload = await response.json();
  startButton.disabled = false;

  if (!response.ok) {
    dependencyStatus.textContent = payload.error || "Could not start match.";
    return;
  }

  setup.hidden = true;
  game.hidden = false;
  state.running = true;
  applyMatchState(payload.state);
  startFrameLoop();
}

function stopFrameLoop() {
  if (state.frameTimer) clearInterval(state.frameTimer);
  state.frameTimer = null;
}

function startFrameLoop() {
  stopFrameLoop();
  state.frameTimer = setInterval(refreshFrameAndState, 100);
  refreshFrameAndState();
}

async function refreshFrameAndState() {
  if (!state.running) return;
  const [stateResponse, frameResponse] = await Promise.all([
    fetch("/api/state"),
    fetch(`/api/frame?t=${Date.now()}`),
  ]);

  if (stateResponse.ok) {
    const payload = await stateResponse.json();
    if (payload.running) applyMatchState(payload.state);
  }

  if (frameResponse.ok) {
    const width = Number(frameResponse.headers.get("X-Canvas-Width"));
    const height = Number(frameResponse.headers.get("X-Canvas-Height"));
    const buffer = await frameResponse.arrayBuffer();
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const rgba = new Uint8ClampedArray(buffer);
    ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  }
}

function applyMatchState(matchState) {
  matchTitle.textContent = `Blue vs ${matchState.opponent.label}`;
  matchMeta.textContent = `${matchState.opponent.architectureLabel}, ${matchState.time}s`;

  if (matchState.finished) {
    state.running = false;
    stopFrameLoop();
  }
}

async function deployAtCanvasPoint(event) {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((event.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.round((event.clientY - rect.top) * (canvas.height / rect.height));
  const response = await fetch("/api/deploy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  });
  if (response.ok) {
    const payload = await response.json();
    applyMatchState(payload.state);
  }
}

opponentSelect.addEventListener("change", updateOpponentMeta);
startButton.addEventListener("click", startMatch);
newMatchButton.addEventListener("click", () => {
  state.running = false;
  stopFrameLoop();
  game.hidden = true;
  setup.hidden = false;
});
canvas.addEventListener("pointerup", deployAtCanvasPoint);
window.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !setup.hidden) startMatch();
});

loadOpponents().catch((error) => {
  dependencyStatus.textContent = error.message;
});
