import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import sphereVert from '../shaders/sphere.vert?raw';
import sphereFrag from '../shaders/sphere.frag?raw';

const STATES = {
    neutral: [1, 0, 0, 0],
    listening: [0, 1, 0, 0],
    thinking: [0, 0, 1, 0],
    speaking: [0, 0, 0, 1],
};

const AgentSphere = ({ currentState = 'neutral', audioData = 0.0 }) => {
    const meshRef = useRef();
    const { viewport } = useThree();

    // Shader Uniforms
    const uniforms = useMemo(
        () => ({
            uTime: { value: 0 },
            uStateWeights: { value: new THREE.Vector4(1, 0, 0, 0) },
            uAudioData: { value: 0 },
            uPixelRatio: { value: Math.min(window.devicePixelRatio, 1.5) }, // Limit DPR to 1.5 for performance
        }),
        []
    );

    // Geometry Generation (8k particles)
    const count = 8000;
    const geometry = useMemo(() => {
        const geo = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const randomVecs = new Float32Array(count * 3);
        const randoms = new Float32Array(count);

        const spherical = new THREE.Spherical();
        const vec3 = new THREE.Vector3();

        for (let i = 0; i < count; i++) {
            const i3 = i * 3;
            // VOLUME Distribution (Dust Cloud)
            vec3.set(
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2
            ).normalize();

            // Random radius (Cube root for uniform volume)
            const radius = Math.cbrt(Math.random()) * 1.2;

            vec3.multiplyScalar(radius);

            positions[i3] = vec3.x;
            positions[i3 + 1] = vec3.y;
            positions[i3 + 2] = vec3.z;

            // Random Noise Vectors
            randomVecs[i3] = (Math.random() - 0.5);
            randomVecs[i3 + 1] = (Math.random() - 0.5);
            randomVecs[i3 + 2] = (Math.random() - 0.5);

            randoms[i] = Math.random();
        }

        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('aRandomVec', new THREE.BufferAttribute(randomVecs, 3));
        geo.setAttribute('aRandom', new THREE.BufferAttribute(randoms, 1));

        geo.computeBoundingSphere();
        return geo;
    }, []);

    // Current blended weights (for smoothing)
    const currentWeights = useRef(new THREE.Vector4(1, 0, 0, 0));

    useFrame((state, delta) => {
        if (!meshRef.current) return;

        // 1. Update Time
        meshRef.current.material.uniforms.uTime.value = state.clock.elapsedTime;

        // 2. Smoothly interpolate Audio Data
        // Simple lerp for audio reactiveness
        meshRef.current.material.uniforms.uAudioData.value = THREE.MathUtils.lerp(
            meshRef.current.material.uniforms.uAudioData.value,
            audioData,
            0.2 // Snappy but smooth
        );

        // 3. Smoothly interpolate State Weights
        const target = STATES[currentState] || STATES.neutral;
        // Lerp each component
        const speed = 4.0 * delta; // Transition speed
        currentWeights.current.x = THREE.MathUtils.lerp(currentWeights.current.x, target[0], speed);
        currentWeights.current.y = THREE.MathUtils.lerp(currentWeights.current.y, target[1], speed);
        currentWeights.current.z = THREE.MathUtils.lerp(currentWeights.current.z, target[2], speed);
        currentWeights.current.w = THREE.MathUtils.lerp(currentWeights.current.w, target[3], speed);

        // Pass to uniform
        meshRef.current.material.uniforms.uStateWeights.value.copy(currentWeights.current);
    });

    return (
        <points ref={meshRef} geometry={geometry}>
            <shaderMaterial
                vertexShader={sphereVert}
                fragmentShader={sphereFrag}
                uniforms={uniforms}
                transparent={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
            />
        </points>
    );
};

export default AgentSphere;
