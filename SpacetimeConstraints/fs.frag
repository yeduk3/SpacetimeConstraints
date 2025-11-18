//
//  fs.frag
//  SpacetimeConstraints
//
//  Created by 이용규 on 11/16/25.
//

#version 410 core

out vec4 out_Color;

uniform int debug;
uniform vec3 debugColor;

void main() {
    if(debug == 1) {
        out_Color = vec4(debugColor, 1);
        return;
    }
    out_Color = vec4(1, 0, 0, 1);
}
