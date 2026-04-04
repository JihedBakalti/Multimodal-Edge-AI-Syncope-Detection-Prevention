# 3D Voice Agent Frontend

A high-fidelity React application rendering the "Stardust Sphere" agent.

## Getting Started

1. Open a terminal in the `frontend` directory:
   ```powershell
   cd frontend
   ```

2. Start the development server:
   ```powershell
   npm run dev
   ```

3. Open the Local URL shown (usually `http://localhost:5173`) in your browser.

## Features

- **Stardust Sphere**: 50,000 particle simulation.
- **GLSL Shaders**: Custom vertex/fragment shaders for high performance.
- **States**:
  - `Neutral`: Gentle breathing motion (Blue).
  - `Listening`: Ripple effect (Green).
  - `Thinking`: Vortex/Tornado animation (Purple).
  - `Speaking`: Pulse/Shockwave synced to audio (Gold).
- **Controls**: Use the control panel (top right) to toggle states and simulate audio levels.

## Tech Stack

- **Vite + React**
- **Three.js** (@react-three/fiber)
- **Post-processing**: Bloom effects.
