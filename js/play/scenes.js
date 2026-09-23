// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/scenes.py: the downstream scenes per robot and the MJCF
// snippet (corridor or maze walls, goal disc, cube) spliced into the robot model
// in place of its <!-- SCENE --> marker. Corridor walls are a port of Isaac's
// CorridorCubePushMixin._corridor_wall_boxes, maze walls of MazeMixin._maze_wall_boxes.
// Keep in lockstep with Python.

const WALL_THICKNESS = 0.2;
const GOAL_MARKER_THICKNESS = 0.02;
// Push scenes: the goal is a line across the corridor, this wide along the corridor.
const GOAL_LINE_WIDTH = 0.15;
const GOAL_COLOR = "0.9 0.1 0.1";          // red disc, same family as the cube and the trails
const WALL_COLOR = "0.84 0.80 0.72";        // warm off-white, for the blue-grey floor
// Robots whose floor is light (the Ant, see robots.js `floorPalette`) pass this instead:
// a pale wall would melt into it. Both are MJCF rgba (linear).
export const GREY_WALL_COLOR = "0.20 0.20 0.21";
const CUBE_COLOR = "0.85 0.12 0.10";        // red, like its trail markers
// A scene can replace the robot's floor through `floorPalette` (a flat colour with
// `line: null` -> no checker, no grid lines; see render.js makeFloorTexture). None does
// at the moment: every G1 scene wears the Ant's grey checker, from robots.js.
// The player's two Ant push scenes use a much lighter cube than Isaac's, where it weighs
// the ant itself (0.9109 kg, shared/scenes.py ANT_MASS): easier to steer by hand.
const ANT_CUBE_MASS = 0.2;
// How far the side walls of a push scene run on past the goal line, so the line sits
// inside the corridor instead of at its open end (corridorWallBoxes `endMargin`).
const PUSH_END_MARGIN = 5.0;
// Every push and maze scene fails after this much simulated time: the Ant's are longer
// (15 m corridors, a 64 m maze route) than the G1's.
const ANT_TASK_TIME_LIMIT = 240;
const G1_TASK_TIME_LIMIT = 180;

function scene(key, title, o = {}) {
  return {
    key, title,
    pathWaypoints: null, corridorWidth: 4.0, wallHeight: 1.0, startMargin: 1.0, endMargin: 0.0,
    mazeMap: null, cellSize: 3.0, shadowExtent: null,
    goalRadius: 0.3, goalTarget: "cube", cubeSize: null, cubePos: [0, 0], cubeMass: 1.0,
    // Corridor scenes only: the goal is a line across the corridor at the last waypoint,
    // wall to wall, and the target succeeds the moment its centre crosses it (instead of
    // a disc of `goalRadius` around the waypoint).
    goalLine: false,
    spawnYaw: 0.0, camera: null, fitView: false, floorPalette: null, description: "",
    // Panel hover card (main.js renderTips): how to actually play this scene, one line
    // per tip. Free text -- edit these. Empty falls back to `description`.
    tips: [],
    // Number key (1-based, as the panel labels them) the scene opens on, so a scene that
    // needs the robot to travel starts on its forward gait instead of skill slot 0.
    // null keeps whatever was picked (robots.js `resetSkill`, else the current choice).
    startKey: null,
    // Seconds of simulated time before the episode counts as a Fail; null never times out.
    timeLimit: null,
    ...o,
    get goalPos() {
      if (this.mazeMap) return mazeGoalPos(this);
      return this.pathWaypoints ? this.pathWaypoints[this.pathWaypoints.length - 1] : null;
    },
    // Unit direction of the corridor's last segment when the goal is a line, else null:
    // the line is normal to it and "past the line" is a positive projection on it.
    get goalDir() {
      return this.goalLine && this.pathWaypoints ? corridorEndDir(this) : null;
    },
  };
}

// ---------------------------------------------------------------------------
// Maze maps (Isaac ant_maze_env.py / g1_maze_env.py, verbatim). Cells sit at odd
// (row, col) positions of the character grid, walls / openings between them, corner
// posts at even/even. S = start (placed at the env origin), G = goal. The first line
// is the far (+y) edge, columns run along +x.
// ---------------------------------------------------------------------------
export const ANT_MAZE_MEDIUM_MAP = [
  "#############",
  "#         #G#",
  "# ######### #",
  "# #   #   # #",
  "# # ### # # #",
  "# #   # #   #",
  "# # # # #####",
  "# # #   # # #",
  "# # ##### # #",
  "# #   #   # #",
  "# ### ### # #",
  "#S          #",
  "#############",
];


export const G1_MAZE_MAP = [
  "###########",
  "#       #G#",
  "# ####### #",
  "# #     # #",
  "# # ### # #",
  "# #   #   #",
  "# # # #####",
  "# # # # # #",
  "# ### # # #",
  "#S        #",
  "###########",
];

export const SCENES = {
  ant: [
    // Same view as the corridor scenes (the default in render.js frameScene), slid 3 m back
    // along +x so the spawn, not the corridor ahead of it, sits in the middle of the frame.
    scene("arena", "Practice", {
      spawnYaw: -Math.PI / 2, camera: { eye: [-9.8, 3.5, 5.85], lookat: [0, 0, 0.6] },
      tips: [
        // Tips for this task go here, one string per line.
      ],
      description: "Open floor. Practice the skills: keys 1-5 pick the skill.",
    }),
    // The default view (render.js frameScene) shifted onto the corridor's centre line
    // (eye.y == target.y == 0): the corridor runs straight away down +x, symmetric in frame.
    scene("push_straight", "Push Cube Straight", {
      pathWaypoints: [[0, 0], [15, 0]], wallHeight: 1.3, goalTarget: "cube", goalLine: true,
      endMargin: PUSH_END_MARGIN,
      cubeSize: 1.0, cubePos: [3, 0], cubeMass: ANT_CUBE_MASS, spawnYaw: -Math.PI / 2,
      camera: { eye: [-6.8, 0, 8.5], lookat: [3.0, 0, 0.6] },
      startKey: 3, timeLimit: ANT_TASK_TIME_LIMIT,
      tips: [
        "Use Skills 1 (Turn Left) and 3 (Move Forward-Right) to position the cube between the front legs.",
        "If your cube gets stuck against the wall, press 5 (Push)!",
        "If your ant accidentally goes past the cube, press 2 (Sharp Turn Left) to return to the front of the cube and try to push it again."
      ],
      description: "Push the cube 15 m down the corridor until its center crosses the red line.",
    }),
    // Framed like the mazes (fitView), as the G1's turn-left corridor is: the L is 17 m on a
    // side, and the close default view showed neither the corner nor the goal.
    scene("push_turn_left", "Push Cube TurnLeft", {
      pathWaypoints: [[0, 0], [15, 0], [15, 15]], wallHeight: 1.3, goalTarget: "cube", goalLine: true,
      endMargin: PUSH_END_MARGIN,
      cubeSize: 1.0, cubePos: [3, 0], cubeMass: ANT_CUBE_MASS, spawnYaw: -Math.PI / 2, fitView: true,
      startKey: 3, timeLimit: ANT_TASK_TIME_LIMIT,
      tips: [
        "Use Skills 1 (Turn Left) and 3 (Move Forward-Right) to position the cube between the front legs.",
        "If your cube gets stuck against the wall, press 5 (Push)!",
        "If your ant accidentally goes past the cube, press 2 (Sharp Turn Left) to return to the front of the cube and try to push it again."
      ],
      description: "Push the cube 15 m, around the corner, and 15 m more across the red line.",
    }),
    // Isaac SingleAnt-Maze-Medium: 4 m cells (the corridor width), 1 m walls. Isaac's own
    // camera is straight down; the player frames every maze itself (render.js _frameBox).
    scene("maze", "Maze", {
      mazeMap: ANT_MAZE_MEDIUM_MAP, cellSize: 4.0, wallHeight: 1.0, goalTarget: "robot", goalRadius: 1.0,
      spawnYaw: -Math.PI / 2,
      startKey: 3, timeLimit: ANT_TASK_TIME_LIMIT,
      tips: [
        "Use Skills 1 (Turn Left) and 3 (Move Forward-Right) to navigate.",
        "When your ant gets stuck on a wall, press 2 (Sharp Turn Left) to escape and reorient the ant toward the desired direction."
      ],
      description: "6×6 maze on a 45° view. Walk to the red disc in the far corner: 64 m shortest route, 11 turns, 5 dead ends.",
    }),
  ],
  g1: [
    // Centred on the spawn, as for the Ant.
    scene("arena", "Practice", {
      camera: { eye: [-9.8, 3.5, 5.85], lookat: [0, 0, 0.6] },
      tips: [
         "How to do a Hard Punch: Press 6 (Lean Backward) to lean back, then press 5!",
      ],
      description: "Open floor. Practice the skills: keys 1-6 pick the skill.",
    }),
    // Twice the corridor of the other G1 scenes: the goal sits 15 m out, so the cube (2 m ahead of
    // the G1, as everywhere) has 13 m to travel instead of 5.5.
    scene("push_straight", "Push Cube Straight", {
      pathWaypoints: [[-2, 0], [15, 0]], wallHeight: 1.5, goalTarget: "cube", goalLine: true,
      endMargin: PUSH_END_MARGIN,
      cubeSize: 1.2, cubePos: [2, 0], cubeMass: 1.5,
      startKey: 2, timeLimit: G1_TASK_TIME_LIMIT,
      tips: [
        // Tips for this task go here, one string per line.
      ],
      description: "Push the 1.2 m cube 13 m down the corridor until its center crosses the red line.",
    }),
    // fitView, as for the Ant: the same view down the corridor's own axis, the whole L in frame.
    scene("push_turn_left", "Push Cube TurnLeft", {
      pathWaypoints: [[0, 0], [7.5, 0], [7.5, 7.5]], wallHeight: 1.5, goalTarget: "cube", goalLine: true,
      endMargin: 2.0,      // half the legs of the other push scenes: 5 m past the goal would dwarf them
      cubeSize: 1.2, cubePos: [2, 0], cubeMass: 1.5, fitView: true,
      startKey: 2, timeLimit: G1_TASK_TIME_LIMIT,
      tips: [
        // Tips for this task go here, one string per line.
      ],
      description: "Push the cube 7.5 m, around the corner, and 7.5 m more across the red line.",
    }),
    // Isaac SingleG1-Maze: 3 m cells and 2 m walls; the player draws the walls at 1.2 m so the
    // 45° view sees over them. Same as the Ant maze: it frames the maze itself rather than
    // copying Isaac's straight-down camera.
    scene("maze", "Maze", {
      mazeMap: G1_MAZE_MAP, cellSize: 3.0, wallHeight: 1.2, goalTarget: "robot", goalRadius: 0.8,
      startKey: 2, timeLimit: G1_TASK_TIME_LIMIT,
      tips: [
        "Try switching back and forth between Skill 1 (Turn Left) and Skill 3 (Turn Right)."
      ],
      description: "5×5 maze on a 45° view. Walk to the red disc in the far corner: 36 m shortest route, 5 turns, 4 dead ends.",
    }),
  ],
  // The Franka has no scene snippet: its MJCF already holds the whole hockey table, so
  // this entry only carries the camera and the label the player shows. The arm is bolted
  // at x = -0.92 on the table centreline facing +x, so the camera looks straight back
  // down that centreline from the far (opponent) end of the table.
  franka: [
    scene("practice", "Practice", {
      camera: { eye: [2.3, 0, 2.0], lookat: [-0.35, 0, 0.92] }, shadowExtent: 2.5,
      tips: [
        "If the robot gets stuck, switching between C (bottom-right) and E (top-right) can sometimes resolve the issue.",
        "If the EEF moves behind the robot, press Reset!",
      ],
      description: "The franka-hockey table with no opponent. Keys Q E S Z C pick the skill; the puck comes back on its own.",
    }),
  ],
};

export function getScene(robot, key) {
  const s = SCENES[robot].find((x) => x.key === key);
  if (!s) throw new Error(`unknown scene ${robot}/${key}`);
  return s;
}

export function corridorWallBoxes(spec) {
  const points = spec.pathWaypoints.map((p) => [+p[0], +p[1]]);
  const halfWidth = 0.5 * spec.corridorWidth, thickness = WALL_THICKNESS;
  const wallOffset = halfWidth + 0.5 * thickness;
  const directions = [];
  for (let i = 0; i + 1 < points.length; i++) {
    const dx = points[i + 1][0] - points[i][0], dy = points[i + 1][1] - points[i][1];
    const len = Math.hypot(dx, dy);
    directions.push([[dx / len, dy / len], len]);
  }
  const cornerExtension = (ci, side) => {
    const inc = directions[ci - 1][0], out = directions[ci][0];
    const turn = inc[0] * out[1] - inc[1] * out[0];
    if (turn === 0) return 0;
    return Math.sign(turn) === side ? -halfWidth : halfWidth + thickness;
  };
  const boxes = [];
  directions.forEach(([dir, len], index) => {
    const start = points[index];
    const nl = [-dir[1], dir[0]];
    for (const [side, sideName] of [[1, "left"], [-1, "right"]]) {
      const startExt = index === 0 ? spec.startMargin : cornerExtension(index, side);
      const endExt = index === directions.length - 1 ? spec.endMargin : cornerExtension(index + 1, side);
      const slab = len + startExt + endExt;
      if (slab <= 0) continue;
      const along = 0.5 * (len + endExt - startExt);
      const center = [
        start[0] + dir[0] * along + nl[0] * side * wallOffset,
        start[1] + dir[1] * along + nl[1] * side * wallOffset,
      ];
      const footprint = dir[0] !== 0 ? [slab, thickness] : [thickness, slab];
      boxes.push({ name: `corridor_wall_${index}_${sideName}`, center, footprint });
    }
  });
  return boxes;
}

// ---------------------------------------------------------------------------
// Maze: port of Isaac's MazeMixin (skill_play_scene.py).
// ---------------------------------------------------------------------------
// Unit direction of a corridor's last segment (the goal line lies across it).
export function corridorEndDir(spec) {
  const p = spec.pathWaypoints, a = p[p.length - 2], b = p[p.length - 1];
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
  return [dx / len, dy / len];
}

// Axis-aligned xy extent of a corridor's walls, the corridor twin of mazeBounds: the
// scenes that ask for `fitView` are framed from it (render.js _frameBox).
export function corridorBounds(spec) {
  const min = [Infinity, Infinity], max = [-Infinity, -Infinity];
  for (const b of corridorWallBoxes(spec)) {
    for (let i = 0; i < 2; i++) {
      min[i] = Math.min(min[i], b.center[i] - 0.5 * b.footprint[i]);
      max[i] = Math.max(max[i], b.center[i] + 0.5 * b.footprint[i]);
    }
  }
  return { min, max };
}

function mazeLines(spec) {
  const lines = spec.mazeMap;
  if (lines.length < 3 || lines.length % 2 === 0 || lines.some((l) => l.length !== lines[0].length)) {
    throw new Error("mazeMap needs an odd number (>= 3) of equally long lines");
  }
  if (lines[0].length < 3 || lines[0].length % 2 === 0) throw new Error("mazeMap lines need an odd length >= 3");
  return lines;
}

function mazeFind(lines, marker) {
  const hits = [];
  lines.forEach((line, row) => { for (let col = 0; col < line.length; col++) if (line[col] === marker) hits.push([row, col]); });
  if (hits.length !== 1) throw new Error(`mazeMap must contain exactly one '${marker}', found ${hits.length}`);
  const [row, col] = hits[0];
  if (row % 2 === 0 || col % 2 === 0) throw new Error(`'${marker}' must sit in a cell (odd row and column), found it at (${row}, ${col})`);
  return hits[0];
}

// World xy (env-local) of a map character: odd indices are cell centers, even ones the
// boundaries between cells; the start cell is the origin.
function mazeCharToXy(spec, lines, row, col) {
  const [startRow, startCol] = mazeFind(lines, "S");
  return [0.5 * (col - startCol) * spec.cellSize, 0.5 * (startRow - row) * spec.cellSize];
}

export function mazeGoalPos(spec) {
  const lines = mazeLines(spec);
  return mazeCharToXy(spec, lines, ...mazeFind(lines, "G"));
}

// Axis-aligned xy extent of the maze's outer wall (for framing the camera / shadows).
export function mazeBounds(spec) {
  const lines = mazeLines(spec);
  const t = WALL_THICKNESS;
  const [x0, y0] = mazeCharToXy(spec, lines, lines.length - 1, 0);
  const [x1, y1] = mazeCharToXy(spec, lines, 0, lines[0].length - 1);
  return { min: [x0 - 0.5 * t, y0 - 0.5 * t], max: [x1 + 0.5 * t, y1 + 0.5 * t] };
}

// Merge every run of '#' along the even rows and even columns into one slab each. Runs are
// extended by half a wall thickness at both ends so the slabs meeting at a corner post
// overlap instead of leaving a notch. A post with no wall on any side (only possible in a map
// with a 2x2 open block) gets its own small block so the map is reproduced exactly.
export function mazeWallBoxes(spec) {
  const lines = mazeLines(spec);
  const thickness = WALL_THICKNESS;
  const covered = new Set();
  const boxes = [];
  const xy = (row, col) => mazeCharToXy(spec, lines, row, col);

  const addRun = (cells, alongX) => {
    if (cells.length < 2) return;   // a lone post; handled below if no run in the other direction covers it
    for (const [r, c] of cells) covered.add(`${r},${c}`);
    const first = xy(...cells[0]), last = xy(...cells[cells.length - 1]);
    const center = [0.5 * (first[0] + last[0]), 0.5 * (first[1] + last[1])];
    const length = (alongX ? Math.abs(last[0] - first[0]) : Math.abs(last[1] - first[1])) + thickness;
    const footprint = alongX ? [length, thickness] : [thickness, length];
    boxes.push({ name: `maze_wall_${boxes.length}`, center, footprint });
  };
  const runs = (indices) => {
    let run = [];
    for (const [row, col] of indices) {
      if (lines[row][col] === "#") run.push([row, col]);
      else { addRun(run, run.length > 1 && run[0][0] === run[1][0]); run = []; }
    }
    addRun(run, run.length > 1 && run[0][0] === run[1][0]);
  };
  for (let row = 0; row < lines.length; row += 2) runs(Array.from({ length: lines[0].length }, (_, col) => [row, col]));
  for (let col = 0; col < lines[0].length; col += 2) runs(Array.from({ length: lines.length }, (_, row) => [row, col]));
  for (let row = 0; row < lines.length; row += 2) {
    for (let col = 0; col < lines[0].length; col += 2) {
      if (lines[row][col] === "#" && !covered.has(`${row},${col}`)) {
        boxes.push({ name: `maze_wall_${boxes.length}`, center: xy(row, col), footprint: [thickness, thickness] });
      }
    }
  }
  return boxes;
}

const f = (x) => Number(x.toPrecision(6)).toString();

function wallGeomXml(b, wallHeight, wallColor = WALL_COLOR) {
  return `<geom name="${b.name}" type="box" pos="${f(b.center[0])} ${f(b.center[1])} ${f(0.5 * wallHeight)}" ` +
    `size="${f(0.5 * b.footprint[0])} ${f(0.5 * b.footprint[1])} ${f(0.5 * wallHeight)}" ` +
    `contype="3" conaffinity="3" condim="3" friction="1 0.5 0.5" rgba="${wallColor} 1"/>`;
}

// The goal marker: a disc of `goalRadius` at the goal, or (goalLine) a thin box across
// the corridor from one wall's inner face to the other's, normal to the last segment.
// Both are named goal_marker: render.js lights them and drops their shadow. Corridors
// are axis-aligned (corridorWallBoxes assumes the same), so the box needs no rotation.
function goalGeomXml(spec) {
  const [gx, gy] = spec.goalPos;
  const pos = `${f(gx)} ${f(gy)} ${f(0.5 * GOAL_MARKER_THICKNESS)}`;
  const common = `contype="0" conaffinity="0" rgba="${GOAL_COLOR} 1"`;
  const dir = spec.goalDir;
  if (dir) {
    const along = 0.5 * GOAL_LINE_WIDTH, across = 0.5 * spec.corridorWidth;
    const [sx, sy] = dir[0] !== 0 ? [along, across] : [across, along];
    return `<geom name="goal_marker" type="box" pos="${pos}" size="${f(sx)} ${f(sy)} ${f(0.5 * GOAL_MARKER_THICKNESS)}" ${common}/>`;
  }
  return `<geom name="goal_marker" type="cylinder" pos="${pos}" size="${f(spec.goalRadius)} ${f(0.5 * GOAL_MARKER_THICKNESS)}" ${common}/>`;
}

export function sceneXml(spec, wallColor = WALL_COLOR) {
  const parts = [];
  if (spec.mazeMap) {
    for (const b of mazeWallBoxes(spec)) parts.push(wallGeomXml(b, spec.wallHeight, wallColor));
    parts.push(goalGeomXml(spec));
  } else if (spec.pathWaypoints) {
    for (const b of corridorWallBoxes(spec)) parts.push(wallGeomXml(b, spec.wallHeight, wallColor));
    parts.push(goalGeomXml(spec));
  }
  if (spec.cubeSize != null) {
    const h = 0.5 * spec.cubeSize;
    parts.push(`<body name="cube" pos="${f(spec.cubePos[0])} ${f(spec.cubePos[1])} ${f(h)}"><freejoint name="cube_root"/>` +
      `<geom name="cube_geom" type="box" size="${f(h)} ${f(h)} ${f(h)}" mass="${f(spec.cubeMass)}" ` +
      `contype="2" conaffinity="1" condim="3" friction="1 0.5 0.5" rgba="${CUBE_COLOR} 1"/></body>`);
  }
  return parts.join("\n    ");
}

// Explicit cube-floor pair: that contact gets the floor's PhysX-averaged friction even
// though MuJoCo would otherwise take the larger (cube) value.
export function contactXml(spec, floorFriction) {
  if (spec.cubeSize == null) return "";
  const fr = f(floorFriction);
  return `<contact><pair geom1="cube_geom" geom2="floor" condim="3" friction="${fr} ${fr} 0.005 0.0001 0.0001"/></contact>`;
}

export function injectScene(robotXml, spec, floorFriction = 1.0, wallColor = WALL_COLOR) {
  for (const marker of ["<!-- SCENE -->", "<!-- CONTACT -->"]) {
    if (!robotXml.includes(marker)) throw new Error(`robot MJCF has no ${marker} marker`);
  }
  return robotXml.replace("<!-- SCENE -->", sceneXml(spec, wallColor))
                 .replace("<!-- CONTACT -->", contactXml(spec, floorFriction));
}
