// sphere.frag
varying vec3 vColor;
varying float vAlpha;

void main() {
    // Coordinate within the point (0.0 to 1.0)
    vec2 pc = gl_PointCoord - vec2(0.5);
    
    // Calculate distance from center of point
    float dist = length(pc);

    // Discard corners to make a circle
    if (dist > 0.5) discard;

    // Create a soft glow (radial gradient)
    // 1.0 at center, fading out
    float glow = 1.0 - (dist * 2.5); // Sharper falloff
    glow = clamp(glow, 0.0, 1.0);
    glow = pow(glow, 3.0); // Sharpen the falloff more

    // Final color with alpha
    gl_FragColor = vec4(vColor, vAlpha * glow);
}
