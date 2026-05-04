import React, { Suspense, useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import AgentSphere from './components/AgentSphere';
import InterSenseSimulator from './components/InterSenseSimulator';

function App() {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8765';
  // WebSocket State
  const [wsState, setWsState] = useState('neutral');
  const [wsAudio, setWsAudio] = useState(0.0);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  const language = 'en';

  // Connect to WebSocket
  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to Medical Assistant Backend');
      setIsConnected(true);
      ws.send(JSON.stringify({ type: 'language', language }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.state) setWsState(data.state);
        if (data.audioData !== undefined) setWsAudio(data.audioData);
      } catch (e) {
        console.error('Error parsing WS message', e);
      }
    };

    ws.onclose = () => setIsConnected(false);
    ws.onerror = (e) => console.error('WS Error:', e);

    return () => ws.close();
  }, [wsUrl]);

  const finalState = wsState;
  const finalAudio = wsAudio;

  return (
    <>
      <Canvas
        camera={{ position: [0, 0, 300], fov: 45 }}
        style={{ background: '#000000' }} // Pure Black for Dust effect
      >
        <ambientLight intensity={0.5} />

        <Suspense fallback={null}>
          <AgentSphere currentState={finalState} audioData={finalAudio} />
        </Suspense>

        <EffectComposer>
          <Bloom
            luminanceThreshold={0.15}
            luminanceSmoothing={0.9}
            height={300}
            intensity={0.8}
          />
        </EffectComposer>

        <OrbitControls
          enableZoom
          enablePan={false}
          minDistance={6.5}
          maxDistance={14}
          target={[0, 0, 0]}
        />
      </Canvas>

      {/* Info Overlay */}
      <div style={{
        position: 'absolute',
        bottom: 20,
        left: 20,
        color: 'white',
        fontFamily: 'sans-serif',
        opacity: 0.7,
        pointerEvents: 'none',
        zIndex: 10
      }}>
        <h2>Medical Assistant</h2>
        <p>State: {(finalState || 'neutral').toUpperCase()}</p>
        <p style={{ fontSize: '0.8em', color: isConnected ? '#4caf50' : '#f44336', marginTop: '5px' }}>
          {isConnected ? '● Connected' : '● Disconnected'}
        </p>
      </div>

      <InterSenseSimulator />
    </>
  );
}

export default App;
