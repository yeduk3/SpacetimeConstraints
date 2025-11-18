//
//  vs.vert
//  SpacetimeConstraints
//
//  Created by 이용규 on 11/16/25.
//

#version 410 core

layout(location = 0) in vec3 currPos;
layout(location = 1) in vec3 nextPos;
layout(location = 2) in vec3 currVel;
layout(location = 3) in vec3 nextVel;

uniform float interTime = 0;

const mat4 cubicHermit = mat4(2, -3, 0, 1, -2, 3, 0, 0, 1, -2, 1, 0, 1, -1, 0, 0);

void main() {
    float t = interTime;
    vec4 tvec = vec4(t*t*t, t*t, t, 1);
//    gl_Position = vec4(mix(currPos, nextPos, interTime), 1); // lineawr
    vec4 pvx = vec4(currPos.x, nextPos.x, currVel.x, -nextVel.x); // cubic hermit
    vec4 pvy = vec4(currPos.y, nextPos.y, currVel.y, -nextVel.y);
    vec4 pvz = vec4(currPos.z, nextPos.z, currVel.z, -nextVel.z);
    float px = dot(tvec, cubicHermit*pvx);
    float py = dot(tvec, cubicHermit*pvy);
    float pz = dot(tvec, cubicHermit*pvz);
    gl_Position = vec4(px, py, pz, 1); // cubic hermit
//    gl_Position = vec4(currPos + (currVel+nextVel)/2.f*t, 1); // physics?
}
