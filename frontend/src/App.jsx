import React, { Suspense, useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { useControls } from 'leva';
import AgentSphere from './components/AgentSphere';
import InterSenseSimulator from './components/InterSenseSimulator';

function App() {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8765';
  // WebSocket State
  const [wsState, setWsState] = useState('neutral');
  const [wsAudio, setWsAudio] = useState(0.0);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  // Language State
  const [language, setLanguage] = useState('en');

  // Connect to WebSocket
  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to Medical Assistant Backend');
      setIsConnected(true);
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

  const handleMicClick = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'command',
        content: '\\voice',
        language: language
      }));
    }
  };

  // Leva Controls (allow manual override if disconnected, or just sync)
  const { currentState, audioData } = useControls({
    currentState: {
      options: ['neutral', 'listening', 'thinking', 'speaking'],
      value: wsState,
      // Update local state if user changes Leva manually (mostly for debug)
      onChange: (v) => { if (!isConnected) setWsState(v); }
    },
    audioData: {
      value: wsAudio,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Audio Level',
      onChange: (v) => { if (!isConnected) setWsAudio(v); }
    }
  });

  // Prioritize WS state if connected
  const finalState = isConnected ? wsState : currentState;
  const finalAudio = isConnected ? wsAudio : audioData;

  return (
    <>
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
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

        <OrbitControls enableZoom={true} enablePan={false} />
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

      {/* Language Selector */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: 20,
        zIndex: 1000,
        display: 'flex',
        gap: '10px'
      }}>
        {['en', 'fr', 'ar', 'tn'].map((lang) => (
          <button
            key={lang}
            onClick={() => setLanguage(lang)}
            style={{
              padding: '8px 16px',
              borderRadius: '20px',
              border: '1px solid rgba(255,255,255,0.2)',
              backgroundColor: language === lang ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.5)',
              color: language === lang ? '#000' : '#fff',
              cursor: 'pointer',
              fontWeight: 'bold',
              backdropFilter: 'blur(5px)',
              transition: 'all 0.2s ease',
              textTransform: 'uppercase',
              fontSize: '0.8rem'
            }}
          >
            {lang}
          </button>
        ))}
      </div>

      <InterSenseSimulator />

      {/* Microphone Trigger Button */}
      <button
        onClick={handleMicClick}
        style={{
          position: 'absolute',
          bottom: 40,
          left: '50%',
          transform: 'translateX(-50%)',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          border: 'none',
          backgroundColor: finalState === 'listening' ? '#ff4444' : 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          color: 'white',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.3s ease',
          boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
          pointerEvents: 'auto',
          zIndex: 1000
        }}
        onMouseEnter={(e) => e.currentTarget.style.transform = 'translateX(-50%) scale(1.1)'}
        onMouseLeave={(e) => e.currentTarget.style.transform = 'translateX(-50%) scale(1.0)'}
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 352 512" width="24" height="24" fill="currentColor">
          <path d="M176 352c53.02 0 96-42.98 96-96V96c0-53.02-42.98-96-96-96S80 42.98 80 96v160c0 53.02 42.98 96 96 96zm160-160h-16c-8.84 0-16 7.16-16 16v48c0 74.8-64.49 134.82-140.79 127.38C96.71 376.89 48 317.11 48 250.3V208c0-8.84-7.16-16-16-16H16c-8.84 0-16 7.16-16 16v40.16c0 89.64 63.97 169.55 152 181.69V464H96c-8.84 0-16 7.16-16 16v16c0 8.84 7.16 16 16 16h160c8.84 0 16-7.16 16-16v-16c0-8.84-7.16-16-16-16h-56v-33.77C285.71 418.47 352 344.9 352 256v-48c0-8.84-7.16-16-16-16z" />
        </svg>
      </button>
    </>
  );
}

export default App;
