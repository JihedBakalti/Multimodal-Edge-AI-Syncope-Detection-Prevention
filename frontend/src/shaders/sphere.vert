// sphere.vert
uniform float uTime;
uniform vec4 uStateWeights; // x: Neutral, y: Listening, z: Thinking, w: Speaking
uniform float uAudioData;
uniform float uPixelRatio;

attribute float aRandom;
attribute vec3 aRandomVec;

varying vec3 vColor;
varying float vAlpha;

// Siri-like Colors
const vec3 COLOR_CYAN = vec3(0.0, 0.9, 1.0);
const vec3 COLOR_MAGENTA = vec3(0.9, 0.1, 1.0);
const vec3 COLOR_BLUE = vec3(0.1, 0.2, 1.0);
const vec3 COLOR_WHITE = vec3(0.9, 0.9, 1.0);

void main() {
    vec3 pos = position; 
    vec3 finalPos = pos;
    float finalSize = 1.0;
    
    // Calculate distance from center for gradients
    float dist = length(pos);
    
    // --- Neutral State (Drifting Dust) ---
    // Particles drift slowly in 3D noise
    vec3 drift = vec3(
        sin(uTime * 0.5 + pos.y + aRandom * 6.0),
        cos(uTime * 0.3 + pos.x + aRandom * 4.0),
        sin(uTime * 0.4 + pos.z + aRandom * 2.0)
    ) * 0.1;
    vec3 neutralPos = pos + drift;
    
    // --- Listening State (Siri Wave) ---
    // Undulating sine waves
    float wave = sin(pos.x * 2.0 + uTime * 3.0) * cos(pos.z * 2.0 + uTime * 2.0) * 0.5;
    vec3 listeningPos = pos + (pos * wave * 0.2) + drift;
    
    // --- Thinking State (Vortex / Galaxy) ---
    // Swirling spiral
    float angle = uTime * 2.0 + dist * 2.0;
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);
    vec2 xz = rot * pos.xz;
    vec3 thinkingPos = vec3(xz.x, pos.y * (0.5 + 0.5 * sin(uTime)), xz.y);
    // Flatten slightly
    thinkingPos.y *= 0.5;
    
    // --- Speaking State (Vibrating Neutral) ---
    // User requested "0.00 lvl brightness" and "slower, not fidgetty"
    // We reduce the frequency significantly to make it a slow sway/pulse instead of a buzz.
    float vibrationBase = 0.3; // Amplitude
    float vibrationIntensity = vibrationBase;
    
    // Low frequency vibration (Slower)
    vec3 vibration = vec3(
        sin(uTime * 3.0 + pos.y * 5.0),
        cos(uTime * 2.5 + pos.x * 6.0),
        sin(uTime * 3.5 + pos.z * 4.0)
    ) * vibrationIntensity * aRandom;
    
    // Combine with the drift from neutral state to keep it "alive"
    vec3 speakingPos = neutralPos + vibration;

    // --- Blend Positions ---
    finalPos = neutralPos * uStateWeights.x +
               listeningPos * uStateWeights.y +
               thinkingPos * uStateWeights.z +
               speakingPos * uStateWeights.w;
               
    // --- Color Mixing (Siri Gradient) ---
    // Mix based on position and randomness
    float gradientMix = (pos.y * 0.2) + 0.5 + (aRandom * 0.2); // 0 to 1
    vec3 baseColor = mix(COLOR_BLUE, COLOR_CYAN, clamp(gradientMix, 0.0, 1.0));
    baseColor = mix(baseColor, COLOR_MAGENTA, clamp(sin(uTime + pos.x)*0.5 + 0.5, 0.0, 1.0));

    // State Colors
    vec3 neutralColor = baseColor * 0.6; // Dimmer
    vec3 listeningColor = mix(COLOR_CYAN, COLOR_WHITE, 0.3);
    vec3 thinkingColor = mix(COLOR_MAGENTA, COLOR_BLUE, 0.5);
    // Speaking color: removed white boost (+vec3(0.2)) and reduced mixing intensity to match "0.00 lvl brightness" request
    // We make it similar to neutral but perhaps just slightly shifted in hue, without extra brightness.
    vec3 speakingColor = mix(COLOR_MAGENTA, COLOR_CYAN, 0.6); 

    vec3 color = neutralColor * uStateWeights.x +
                 listeningColor * uStateWeights.y +
                 thinkingColor * uStateWeights.z +
                 speakingColor * uStateWeights.w;

    // --- Size Calculation ---
    // Distance attenuation
    vec4 mvPosition = modelViewMatrix * vec4(finalPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Smaller particles for "Dust" look
    // Base size 15.0 -> 8.0 -> 18.0 (compensate for low count)
    // Remove audio influence on size for speaking, keep it for others if needed or just remove globally if audio is broken
    // User asked "stay the same", implying the vibration part mainly. 
    // But "not be audio lvl dynamic" suggests removing uAudioData usage.
    // Let's make size constant but slightly larger for speaking.
    float sizeMultiplier = 1.0 + (uStateWeights.w * 0.5); // 1.5x size when speaking
    finalSize = 18.0 * uPixelRatio * sizeMultiplier; 
    
    // Randomize size slightly
    finalSize *= (0.5 + aRandom * 1.0);

    gl_PointSize = finalSize * (4.0 / -mvPosition.z);

    vColor = color;
    // Lower alpha for dust effect, constant high alpha for speaking
    vAlpha = 0.6 + (0.4 * uStateWeights.w); 
}
