// Wood step ladder parametric OpenSCAD model
// Units: meters

total_h = 1.6;
front_run = 1.04;
base_h = 0.16;
base_w = 0.64;
base_l = 1.04;
step_count = 7;
step_w = 0.64;
step_d = 0.13;
step_t = 0.03;
step_top_spacing = 0.23;
step_y_spacing = 0.135;
rear_spread = 1.04;
side_t = 0.06;
side_d = 0.14;
top_d = 0.13;
top_t = 0.03;
rear_beam_h = 0.72;
top_z = base_h + total_h;

module box(size) { cube(size, center=true); }

function y_from_z(z) = -(z - base_h) * front_run / total_h;
function step_top_z(i) = base_h + step_t + i * step_top_spacing;
function step_front_y(i) = -i * step_y_spacing;

translate([0, -base_l / 2, base_h / 2])
    box([base_w, base_l, base_h]);

for (i = [0:step_count - 1]) {
    step_z = step_top_z(i);
    center_z = step_z - step_t / 2;
    translate([0, step_front_y(i) - step_d / 2, center_z])
        box([step_w, step_d, step_t]);
}

front_phi = atan2(total_h, front_run);
rail_len = sqrt(pow(total_h, 2) + pow(front_run, 2)) + 0.4;
rail_mid_y = y_from_z(top_z) / 2;
rail_mid_z = (base_h + top_z) / 2;

for (x = [-step_w/2 - side_t/2, step_w/2 + side_t/2]) {
    translate([x, rail_mid_y, rail_mid_z])
        rotate([90-front_phi, 0, 0])
            box([side_t, side_d, rail_len]);
}

top_y = y_from_z(top_z);
translate([0, top_y - top_d * 0.15, top_z - top_t / 2])
    box([step_w + 0.12, top_d, top_t]);

for (x = [-step_w/2 - side_t/2, step_w/2 + side_t/2]) {
    translate([x, top_y, top_z + 0.11])
        box([0.09, 0.12, 0.22]);
}

rear_y = y_from_z(base_h) - rear_spread;
rear_mid_y = (rear_y + top_y) / 2;
rear_mid_z = (0.02 + top_z) / 2;
rear_angle = atan2(top_z - 0.02, top_y - rear_y);

for (x = [-step_w/2 - 0.05, step_w/2 + 0.05]) {
    translate([x, rear_mid_y, rear_mid_z])
        rotate([90-rear_angle, 0, 0])
            box([0.08, 0.08, (top_z - 0.02) / sin(rear_angle) + 0.16]);
}

translate([0, rear_y + (top_y - rear_y) * (rear_beam_h / total_h), rear_beam_h])
    box([step_w + 0.12, 0.07, 0.08]);

// Visual-only measurement triangle: green run, blue height, red hypotenuse.
translate([step_w / 2 + side_t + 0.08, -front_run / 2, base_h])
    color("green") box([0.018, front_run, 0.018]);
translate([step_w / 2 + side_t + 0.08, -front_run, base_h + total_h / 2])
    color("blue") box([0.018, 0.018, total_h]);
translate([step_w / 2 + side_t + 0.08, -front_run / 2, base_h + total_h / 2])
    rotate([90-front_phi, 0, 0])
        color("red") box([0.018, 0.018, sqrt(pow(total_h, 2) + pow(front_run, 2))]);
