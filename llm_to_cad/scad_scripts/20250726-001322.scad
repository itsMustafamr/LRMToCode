difference() {
    union() {
        cylinder(h = 30, r = 10);
        translate([0, 0, 30]) cube([20, 20, 10], center = true);
    }
    translate([-5, -5, -5]) cube([10, 10, 40]);
}